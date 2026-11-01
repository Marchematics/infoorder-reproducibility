import contextlib
import json
import os
import time
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
from reader_protocol import reasoned_reader_prompt, parse_final_answer

WRITER_SYSTEM = 'Store facts and links that new evidence may need. Preserve unresolved observations, exact names, bindings, and source identities. Output compact factual memory only. The task and its rules are always available to the reader; devote the memory to evidence.'
SUMMARY_SYSTEM = 'Summarize the evidence concisely for the stated task. Output the summary only.'
BRIDGE_SYSTEM = 'Preserve facts and relations that may be needed to connect the current evidence to later evidence, including unresolved names, identifiers, sources, and constraints. Output a concise memory only.'
EXTRACT_SYSTEM = 'Extract the most useful factual statements verbatim, keeping their entity and source names. Output the extracted facts only.'
STRUCTURE_SYSTEM = 'Store useful evidence as compact named records and relations. Preserve entity bindings and source identifiers. Output records only.'

def writer_prompt(question, old_memory, evidence, budget=None):
    limit=f'Keep the entire returned memory within {budget} tokens.\n\n' if budget is not None else ''
    return limit+f'Fixed task:\n{question}\n\nPrevious memory:\n{old_memory}\n\nNew evidence:\n{evidence}'

def reader_prompt(question, memory, later=''):
    return f'Task:\n{question}\n\nStored evidence:\n{memory}\n\nNew evidence:\n{later}'

class MemoryModel:
    def __init__(self, model_path, adapter=None, train=False, seed=17):
        self.path = Path(model_path)
        manifest = self.path/'infoorder_manifest.json'
        self.provenance = json.loads(manifest.read_text()) if manifest.exists() else {'model': self.path.name}
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        cls = AutoModelForImageTextToText if config.model_type == 'qwen3_5' else AutoModelForCausalLM
        phi_attention = os.environ.get('INFOORDER_PHI_ATTENTION', 'sdpa')
        if phi_attention not in ('sdpa', 'flash_causal'):
            raise ValueError('INFOORDER_PHI_ATTENTION must be sdpa or flash_causal')
        self.attention_backend = 'sdpa'
        if config.model_type == 'phi3' and phi_attention == 'flash_causal':
            from phi_attention import register
            self.attention_backend = register()
        selected=os.environ.get('INFOORDER_DEVICE') or ('cuda:0' if torch.cuda.is_available() else
                 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.cuda_memory_cap_mib = int(os.environ.get('INFOORDER_CUDA_MEMORY_CAP_MIB', '0'))
        if self.cuda_memory_cap_mib < 0:
            raise ValueError('CUDA process memory cap must be nonnegative')
        if self.cuda_memory_cap_mib:
            if not selected.startswith('cuda'):
                raise ValueError('A CUDA memory cap requires a CUDA device')
            total = torch.cuda.get_device_properties(selected).total_memory
            fraction = self.cuda_memory_cap_mib * 1024**2 / total
            if fraction > 1:
                raise ValueError('CUDA memory cap exceeds device capacity')
            torch.cuda.set_per_process_memory_fraction(fraction, selected)
                                                                          
                                                                            
        if selected=='mps':os.environ.setdefault('HF_DEACTIVATE_ASYNC_LOAD','1')
        self.model = cls.from_pretrained(model_path, dtype=torch.bfloat16 if selected!='cpu' else torch.float32,
                        device_map={'':selected},
                        local_files_only=True, trust_remote_code=False, attn_implementation=self.attention_backend)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=train)
        elif train:
            from peft import LoraConfig, get_peft_model
            projections={'q_proj','k_proj','v_proj','o_proj','qkv_proj','in_proj_qkv',
                         'in_proj_z','in_proj_a','in_proj_b','out_proj'}
            targets=sorted({name.rsplit('.',1)[-1] for name,module in self.model.named_modules()
                            if name.rsplit('.',1)[-1] in projections and 'visual' not in name})
            self.model = get_peft_model(self.model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                       target_modules=targets, task_type='CAUSAL_LM'))
        if train:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.model.eval()
        self.calls = []
        self.reader_scoring = os.environ.get('INFOORDER_READER_SCORING',
                                             'scalar' if selected == 'mps' else 'batch')
        if self.reader_scoring not in ('scalar', 'batch'):
            raise ValueError('INFOORDER_READER_SCORING must be scalar or batch')
        self.prefill_chunk_size = int(os.environ.get('INFOORDER_PREFILL_CHUNK_SIZE', '8192'))
        if self.prefill_chunk_size < 0:
            raise ValueError('INFOORDER_PREFILL_CHUNK_SIZE must be nonnegative')
                                                                           
                                                                            
        rope = getattr(config, 'rope_parameters', None) or {}
        if (config.model_type == 'phi3' and rope.get('rope_type') == 'longrope'
                and self.prefill_chunk_size
                and self.prefill_chunk_size <= rope['original_max_position_embeddings']):
            raise ValueError('Phi long-context prefill must start beyond its short-context range')

    @property
    def device(self):
        return next(self.model.parameters()).device

    def prompt(self, system, user):
        return self.tokenizer.apply_chat_template([{'role':'system','content':system}, {'role':'user','content':user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def reader_context(self):
        return self.model.disable_adapter() if hasattr(self.model, 'disable_adapter') else contextlib.nullcontext()

    def _record(self, kind, tokens, output_tokens, elapsed):
        self.calls.append({'kind':kind,'input_tokens':tokens,'output_tokens':output_tokens,'seconds':elapsed})

    @torch.no_grad()
    def generate(self, system, user, budget=128, sample=False, reader=False):
        prompt = self.prompt(system, user)
        tokens = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        start = time.perf_counter()
        self.model.eval()
        chunk_size = (self.prefill_chunk_size if self.prefill_chunk_size
                      and tokens.input_ids.shape[1] > self.prefill_chunk_size else None)
        with self.reader_context() if reader else contextlib.nullcontext():
            outputs = self.model.generate(**tokens, max_new_tokens=budget, do_sample=sample,
                        **({'temperature':0.8, 'top_p':0.95} if sample else {}),
                        pad_token_id=self.tokenizer.pad_token_id, use_cache=True,
                        prefill_chunk_size=chunk_size)
        tail = outputs[0, tokens.input_ids.shape[1]:]
        text = self.tokenizer.decode(tail, skip_special_tokens=True).strip()
                                                                       
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:budget]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        self._record('reader_generate' if reader else 'writer_generate', tokens.input_ids.shape[1], len(tail), time.perf_counter()-start)
        self.calls[-1]['prefill_chunk_size'] = chunk_size
        return text

    @torch.no_grad()
    def binary_prediction(self, question, memory, later=''):
        prompt = self.prompt('Use the supplied evidence to answer. Return only 0 or 1.', reader_prompt(question,memory,later))
        tokens = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        labels = [self.tokenizer.encode(str(i), add_special_tokens=False) for i in (0,1)]
        start = time.perf_counter()
        with self.reader_context():
            if all(len(x)==1 for x in labels):
                logits = self.model(**tokens, use_cache=False, logits_to_keep=1).logits[0,-1,:].float()
                probs = torch.softmax(logits[[x[0] for x in labels]],dim=0)
            else:
                scores = [self.completion_logprob(prompt, str(i), reader=True).detach() for i in (0,1)]
                probs = torch.softmax(torch.stack(scores),dim=0)
        self._record('reader_score',tokens.input_ids.shape[1],0,time.perf_counter()-start)
        return probs.cpu().tolist()

    @torch.no_grad()
    def binary_predictions(self, requests, batch_size=8):
        """Score independent requests without carrying a cache between them."""
                                                                              
                                                                                
                                         
        if self.device.type=='mps' or self.reader_scoring=='scalar':
            return [self.binary_prediction(*request) for request in requests]
        labels = [self.tokenizer.encode(str(i), add_special_tokens=False) for i in (0, 1)]
        if not all(len(x) == 1 for x in labels):
            return [self.binary_prediction(*request) for request in requests]
        answers = []
        for offset in range(0, len(requests), batch_size):
            batch = requests[offset:offset + batch_size]
            prompts = [self.prompt('Use the supplied evidence to answer. Return only 0 or 1.',
                                   reader_prompt(*request)) for request in batch]
                                                                              
            original_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = 'right'
            try:
                tokens = self.tokenizer(prompts, padding=True, return_tensors='pt').to(self.device)
            finally:
                self.tokenizer.padding_side = original_side
            start = time.perf_counter()
            last = tokens.attention_mask.sum(dim=1) - 1
            positions = torch.unique(last, sorted=True)
            with self.reader_context():
                logits = self.model(**tokens, use_cache=False, logits_to_keep=positions).logits
                local_positions = torch.searchsorted(positions,last)
                selected = logits[torch.arange(len(batch), device=self.device), local_positions][:, [x[0] for x in labels]].float()
                answers.extend(torch.softmax(selected, dim=1).cpu().tolist())
            self._record('reader_score_batch', int(tokens.attention_mask.sum()), 0, time.perf_counter() - start)
            self.calls[-1].update(requests=len(batch), padded_input_tokens=tokens.input_ids.numel())
        return answers

    def completion_logprob(self, prompt, completion, reader=False):
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if not answer_ids:
            raise ValueError('Empty completion is not a valid preference')
        ids = torch.tensor([prompt_ids+answer_ids],device=self.device)
        with self.reader_context() if reader else contextlib.nullcontext():
            logits = self.model(input_ids=ids,use_cache=False,logits_to_keep=len(answer_ids)+1).logits[:,:-1,:].float()
        target = torch.tensor([answer_ids],device=self.device)
        return torch.log_softmax(logits,dim=-1).gather(-1,target.unsqueeze(-1)).squeeze(-1).sum()

    @torch.no_grad()
    def candidate_scores(self, system, user, candidates):
        """Frozen-reader likelihoods for a task's public allowed output set."""
        self.model.eval()
        prompt=self.prompt(system,user)
        start=time.perf_counter()
        scores=[float(self.completion_logprob(prompt,text,reader=True)) for text in candidates]
        probabilities=torch.softmax(torch.tensor(scores,dtype=torch.float64),dim=0).tolist()
        prompt_tokens=len(self.tokenizer.encode(prompt,add_special_tokens=False))
        candidate_tokens=sum(len(self.tokenizer.encode(text,add_special_tokens=False)) for text in candidates)
        self._record('reader_candidate_score',len(candidates)*prompt_tokens+candidate_tokens,
                     0,time.perf_counter()-start)
        self.calls[-1]['candidates']=len(candidates)
        return scores,probabilities

    def update(self, question, memory, evidence, budget, system=WRITER_SYSTEM, sample=False):
        return self.generate(system,writer_prompt(question,memory,evidence,budget),budget,sample)

    def answer(self, question, memory, later='', budget=64):
        return self.generate('Answer the task using the supplied evidence. Give a concise final answer only.',
                             f'Stored evidence:\n{memory}\n\nNew evidence:\n{later}\n\nTask:\n{question}',
                             budget,reader=True)

    def reasoned_answer(self, question, memory, later='', binary=False, budget=192):
        system, user = reasoned_reader_prompt(question, memory, later, binary)
        raw = self.generate(system, user, budget, reader=True)
        answer = parse_final_answer(raw, binary)
        return {'raw': raw, 'answer': answer, 'valid': answer is not None}

    @torch.no_grad()
    def staged_binary_scores(self, system, user):
        prompt = self.prompt(system, user) + 'Final answer: '
        start = time.perf_counter()
        scores = [self.completion_logprob(prompt, value, reader=True) for value in ('0', '1')]
        if not all(torch.isfinite(value) for value in scores):
            raise ValueError('Nonfinite staged binary likelihood')
        probabilities = torch.softmax(torch.stack(scores).double(), dim=0).cpu().tolist()
        tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        alternatives = sum(len(self.tokenizer.encode(value, add_special_tokens=False)) for value in ('0', '1'))
        self._record('reader_staged_binary', 2 * tokens + alternatives, 0, time.perf_counter() - start)
        self.calls[-1]['requests'] = 2
        return [float(value) for value in scores], probabilities

    def save_usage(self, target):
        Path(target).write_text(json.dumps({'model':self.provenance,'calls':self.calls},indent=2))
