import tensorflow as tf
import os
import numpy as np
import argparse
import modeling_gpt2
import torch
import sentencepiece as spm
from torch.nn import functional as F
from tqdm import trange

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, context, length, n_ctx, tokenizer, temperature=1.0, top_k=30, top_p=0.0,
                         repitition_penalty=1.2,
                         device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'inputs':tf.convert_to_tensor(generated[0][-(n_ctx - 1):].unsqueeze(0).numpy())}
            outputs = model(
                **inputs)
            next_token_logits = outputs[0, -1, :]
            next_token_logits = torch.from_numpy(next_token_logits.numpy())
            for id in set(generated):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated.tolist()[0]



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=50, type=int, required=False, help='生成长度')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=5, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0.95, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='configs/gpt2/model_config_small.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--spm_model_path', default='spm_model/ch.model', type=str, required=False, help='')
    parser.add_argument('--model_path', default='model/', type=str, required=False, help='模型路径')
    parser.add_argument('--prefix', default='丨', type=str, required=False, help='生成文章的开头')
    parser.add_argument('--repetition_penalty', default=1.1, type=float, required=False)
    parser.add_argument('--n_ctx', default=512, type=int)
    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    model = modeling_gpt2.TFGPT2LMHeadModel.from_pretrained(args.model_path)

    length = args.length
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty
    n_ctx = args.n_ctx

    ch_sp = spm.SentencePieceProcessor()
    ch_sp.Load(args.spm_model_path)

    while True:
        context = ch_sp.encode_as_ids(args.prefix)
        generated = sample_sequence(model, context, length, n_ctx, ch_sp, temperature=temperature, top_k=topk, top_p=topp,
                        repitition_penalty=repetition_penalty)
        print(ch_sp.decode_ids(generated))

if __name__ == '__main__':
    main()
