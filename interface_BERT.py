import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer

import numpy as np
from tqdm import tqdm

def tokenize_sequence(tokenizer, sequence):
    sequence = sequence.split(" \|\|\| ")
    if len(sequence) == 1:
        sentence_a = sequence[0]
        sentence_b = ""
    else:
        sentence_a = sequence[0]
        sentence_b = sequence[1]

    tokens_a = tokenizer.tokenize(sentence_a)
    tokens_b = tokenizer.tokenize(sentence_b)
    tokens_a_delim = ['[CLS]'] + tokens_a + ['[SEP]']
    tokens_b_delim = tokens_b + (['[SEP]'] if len(tokens_b) > 0 else [])
    token_ids = tokenizer.convert_tokens_to_ids(tokens_a_delim + tokens_b_delim)
    tokens_tensor = torch.tensor([token_ids])
    token_type_tensor = torch.LongTensor([[0] * len(tokens_a_delim) + [1] * len(tokens_b_delim)])

    return tokens_a_delim, tokens_b_delim, tokens_tensor, token_type_tensor


def apply_bert_get_attention(model, tokenizer, sequence):
    """
    Essentially isolated from jessevig/bertviz
    :param model: bert
    :param tokenizer: bert tokenizer
    :param sequence: single sentence
    :return:
    """

    model.eval()
    tokens_a, tokens_b, tokens_tensor, token_type_tensor = tokenize_sequence(tokenizer, sequence)

    if next(model.parameters()).is_cuda:
        tokens_tensor = tokens_tensor.cuda()
        token_type_tensor = token_type_tensor.cuda()

    _, _, attn_data_list = model(tokens_tensor, token_type_ids=token_type_tensor)
    attn_tensor = torch.stack([attn_data['attn_probs'] for attn_data in attn_data_list])
    attn = attn_tensor.data.cpu().numpy()

    return tokens_a, tokens_b, attn


def apply_bert(items, tokenizer, args):
    print("Applying BERT.")

    model = BertModel.from_pretrained(args.bert)

    if args.cuda:
        model.cuda()

    data_for_all_items = []
    for _, each_item in tqdm(items.iterrows(), total=len(items)):

        if args.method == "attention":
            tokens_a, tokens_b, attention = apply_bert_get_attention(model, tokenizer, each_item['sentence'])
            attention = attention.squeeze()
            if args.combine == "chain":
                weights_per_layer = compute_pMAT(attention, layer_norm=args.normalize_heads)
            else:
                weights_per_layer = compute_MAT(attention, layer_norm=args.normalize_heads)
            ## Nope, instead of the following, transpose the gradients: from input_token to output_token
            # weights_per_layer = weights_per_layer.transpose(0,2,1)  # for uniformity with gradients: (layer, output_token, input_token)
        elif args.method == "gradient":
            tokens_a, tokens_b, weights_per_layer = apply_bert_get_gradients(model, tokenizer, each_item['sentence'], chain=args.combine=="chain")
            weights_per_layer = weights_per_layer.transpose(0,2,1)  # for uniformity with attention weights: (layer, input_token, output_token)
            # TODO IMPORTANT Not sure if this is right; the picture comes out all weird, almost the inverse of attention-based...

        data_for_all_items.append(weights_per_layer)

    return data_for_all_items


def apply_bert_get_gradients(model, tokenizer, sequence, chain):
    """
    :param model: bert
    :param tokenizer: bert tokenizer
    :param sequence: single sentence
    :param chain: whether to compute the gradients all the way back, i.e., wrt the input embeddings
    :return:
    """
    model.train()   # Because I need the gradients
    model.zero_grad()

    tokens_a, tokens_b, tokens_tensor, token_type_tensor = tokenize_sequence(tokenizer, sequence)

    if next(model.parameters()).is_cuda:
        tokens_tensor = tokens_tensor.cuda()
        token_type_tensor = token_type_tensor.cuda()

    encoded_layers, _, _, embedding_output = model(tokens_tensor, token_type_ids=token_type_tensor, output_embedding=True)
#   [n_layers x [batch_size, seq_len, hidden]]      [seq_len x hidden]

    previous_activations = embedding_output
    gradients = []
    for layer in encoded_layers:        # layer: [batch_size, seq_len, hidden]
        target = embedding_output if chain else previous_activations
        gradients_for_layer = []

        for token_idx in range(layer.shape[1]): # loop over output tokens
            target.retain_grad()    # not sure if needed every iteration
            mask = torch.zeros_like(layer)
            mask[:,token_idx,:] = 1
            layer.backward(mask, retain_graph=True)
            gradient = target.grad.data    # [batch_size, seq_len, hidden]
            gradient = gradient.squeeze().clone().cpu().numpy()

            # TODO Ideally this would be done still on cuda
            gradient_norm = np.linalg.norm(gradient, axis=-1)   # take norm per input token
            gradients_for_layer.append(gradient_norm)
            target.grad.data.zero_()
            previous_activations = layer

        gradients_for_layer = np.stack(gradients_for_layer)
        gradients.append(gradients_for_layer)

    gradients = np.stack(gradients)

    return tokens_a, tokens_b, gradients


def normalize(v):
    """
    Divides a vector by its norm.
    :param v:
    :return:
    """
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

# TODO Merge the following two functions and change names
def compute_MAT(all_attention_weights, layer_norm=True):
    """
    Computes Mean Attention per Token (MAT), i.e,, mean across all attention heads, per layer.
    :param all_attention_weights: as retrieved from attention visualizer
    :param layer_norm: whether to normalize
    :return: mean attention weights (across heads) per layer
    """
    # TODO Ideally this would be done still on cuda
    mean_activations_per_layer = []
    for heads_of_layer in all_attention_weights:
        summed_activations = np.zeros_like(all_attention_weights[0][1])
        for head in heads_of_layer:      # n_tokens × n_tokens
            activations_per_head = head.copy().transpose()
            # (i,j) = how much (activations coming from) token i influences token j
            if layer_norm:       # Normalize influence (across all tokens i) on each token j
                for j in range(0, len(activations_per_head)):
                    activations_per_head[:,j] = normalize(activations_per_head[:, j])   # TODO check if this really makes sense... I don't think it does.
            summed_activations += activations_per_head

        mean_activations_per_layer.append(summed_activations / all_attention_weights.shape[1])

    return np.stack(mean_activations_per_layer)


def compute_pMAT(all_attention_weights, layer_norm=True):
    """
    Computes Percolated Mean Attention per Token (pMAT), through all layers.
    :param all_attention_weights: as retrieved from the attention visualizer
    :param layer_norm: whether to normalize the weights of each attention head
    :return: percolated activations up to every layer
    """
    # TODO Ideally this would be done still on cuda
    # TODO: Think about this. What about normalizing per layer, instead of per head? Does that make any sense? Yes, a bit. However, since BERT has LAYER NORM in each attention head, outputs of all heads will have same mean/variance. Does this mean that all heads will contribute same amount of information? Yes, roughly.
    percolated_activations_per_layer = []
    percolated_activations = np.diag(np.ones(all_attention_weights.shape[-1]))      # n_tokens × n_tokens
    for layer in all_attention_weights:
        summed_activations = np.zeros_like(percolated_activations)
        for head in layer:      # n_tokens × n_tokens
            head_t = head.copy().transpose()    # TODO Check if correct
            activations_per_head = np.matmul(head_t, percolated_activations)
            # (i,j) = how much (activations coming ultimately from) token i influences token j
            if layer_norm:       # Normalize influence (across all tokens i) on each token j
                for j in range(0, len(activations_per_head)):
                    activations_per_head[:,j] = normalize(activations_per_head[:, j])       # TODO Check if this makes sense
            summed_activations += activations_per_head
        # normalize or things get out of hand
        summed_activations = normalize(summed_activations)  # TODO Check how this relates to layernorm

        # for the next layer, use summed_activations as the next input activations
        percolated_activations = summed_activations
        # I believe normalizing the activations (as a whole or per col) makes no difference.

        percolated_activations_per_layer.append(percolated_activations)

    return np.stack(percolated_activations_per_layer)
