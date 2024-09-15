
import torch


def broadcast_all(tensors, exclude_dims=()):
    # dim: dim_th axis will not be broadcasted

    shape_tensor = torch.stack([torch.tensor(t.shape) for t in tensors], dim=0)
    to_shape = shape_tensor.max(dim=0).values.tolist()
    new_tensors = []

    for t in tensors:
        for dim in exclude_dims:
            to_shape[dim] = t.shape[dim]
        new_tensors.append(t.expand(*to_shape))

    return new_tensors


def broadcast_concat(tensors, dim):
    """
    Concatenate tensors with broadcasting
    """
    new_tensors = broadcast_all(tensors, exclude_dims=(dim,))

    return torch.cat(new_tensors, dim=dim)


def broadcast_stack(tensors, dim):
    """
    Stack tensors with broadcasting
    """
    new_tensors = broadcast_all(tensors)
    return torch.stack(new_tensors, dim=dim)


# def isolate_patterns(p):
#     snippets = re.split(r"\[|\]", p)

#     inside_brackets = snippets[1].strip()
#     outside_brackets = " ".join([snippets[0].strip(), snippets[2].strip()])

#     n_match = re.search(r"\s(\w+)\s*\[", p)

#     select_axis = n_match.group(1) if n_match else ""

#     return inside_brackets, outside_brackets, select_axis


# def reshape_patterns(tensor_pattern, indices_pattern, select_axis):
#     parsed_tensor_pattern = tensor_pattern.split()
#     parsed_indices_pattern = indices_pattern.split()

#     common_axis = list(set(parsed_tensor_pattern) & set(parsed_indices_pattern))

#     new_tensor_pattern = copy.copy(common_axis)
#     new_tensor_pattern.append(select_axis)

#     for ax in parsed_tensor_pattern:
#         if ax not in common_axis and ax != select_axis:
#             new_tensor_pattern.append(ax)

#     # new_select_pattern = copy.copy(common_axis)
#     # for ax in parsed_select_pattern:
#     #     if ax not in common_axis:
#     #         new_select_pattern.append(ax)
#     # new_select_pattern = " ".join(new_select_pattern)

#     new_tensor_pattern = " ".join(new_tensor_pattern)

#     return new_tensor_pattern, common_axis


# def get_axis_dim(tensor, pattern):
#     ret_dict = {}
#     for i, ax in enumerate(pattern.split()):
#         ret_dict[ax] = tensor.shape[i]

#     return ret_dict


# def gather_general(tensor, indices, pattern):
#     """
#     This is an extended version of torch.gather,
#     For original torch.gather, it only supports the case where "tensor.ndim == indices.ndim"
#     Then depending on `dim`, it gathers the value from dim-th axis

#     However, suppose we have a case below
#     * tensor (M, N, D)
#     * indices (M, K, L)
#     * outputs (M, K, L, D)
#     where tensor[m, indices[m, k, l], :] = outputs[m, k, l, :]

#     It differs for original cases as follows
#     * tensor.ndim != indices.ndim
#     * position of axis are flexible at both sides
#       (e.g. for original, dimensions are fixed, when dim=2 and both ndim=3, output[i][j][k] = input[i][j][indices[i][j][k]])
#     * allow for extra axis in indices (e.g. K in this case)
#       these extra axis will act similar as batch axis for the output tensor
#     * allow for indices to ignore some axis from tensor (e.g. D in this case)
#       these axes will expand to a hyperplane, which means we are gathering "hyperplane" instead of "value"

#     The output dimension must be union of (1) common axis from both tensor and indices (2) extra axis from indices
#     We define the notation in light of einops
#     ```
#     m n[m k l] d -> m k l d
#     ```
#     Position of [] stands for gathered axis, common axis are similar to original gather, and extra axis are appended at the end
#     """
#     groups = re.match(r"(.*)->(.*)", pattern)
#     _tensor_pattern = groups[1].strip()
#     output_pattern = groups[2].strip()
#     device = tensor.device
#     indices_flat = indices.reshape(-1)

#     indices_pattern, tensor_pattern, select_axis = isolate_patterns(_tensor_pattern)
#     # dim_per_axis_tensor = get_axis_dim(tensor, tensor_pattern)
#     dim_per_axis_indices = get_axis_dim(indices, indices_pattern)

#     new_tensor_pattern, common_axis = reshape_patterns(
#         tensor_pattern=tensor_pattern,
#         indices_pattern=indices_pattern,
#         select_axis=select_axis,
#     )

#     reshaped_tensor = rearrange(tensor, f"{tensor_pattern} -> {new_tensor_pattern}")

#     def get_pad_indices(ax):
#         pad_ind = repeat(
#             torch.arange(dim_per_axis_indices[ax], device=device),
#             f"{ax} -> {indices_pattern}",
#             **dim_per_axis_indices,
#         ).reshape(-1)
#         return pad_ind

#     select_indices = [get_pad_indices(ax) for ax in common_axis]
#     select_indices += [indices_flat]

#     selected_tensor = reshaped_tensor[select_indices]
#     parsed_indices_pattern = indices_pattern.split()
#     selected_pattern = f"({indices_pattern}) " \
#         + " ".join([ax for ax in new_tensor_pattern.split() \
#             if ax not in parsed_indices_pattern and ax != select_axis])

#     selected_tensor = rearrange(
#         selected_tensor,
#         f"{selected_pattern} -> {output_pattern}",
#         **dim_per_axis_indices,
#     )
#     return selected_tensor


# def test_gather_general():
#     M = 300
#     K = 100
#     L = 200
#     N = 100
#     D = 400

#     tensor = torch.arange(M * N * D).reshape(M, N, D).cuda()
#     indices = torch.randint(0, N, (M, K, L)).cuda()
#     # m = 2, n = 3, k = 2, l = 3, d = 4
#     # select X[m, I[k, l], d] given k, l, m
#     pattern = "m n[m k l] d -> k l m d"

#     ret = gather_general(tensor, indices, pattern=pattern)

#     def check_by_index(k, l, m):
#         return (ret[k, l, m, :] == tensor[m, indices[m, k, l], :]).all().item()

#     for t in range(100):
#         k = np.random.randint(0, K)
#         l = np.random.randint(0, L)
#         m = np.random.randint(0, M)
#         check_res = check_by_index(k, l, m)
#         if not check_res:
#             print(f"Fail at {k, l, m}")
#             break
#         print(f"Pass at {k, l, m}")


# if __name__ == "__main__":
#     test_gather_general()
