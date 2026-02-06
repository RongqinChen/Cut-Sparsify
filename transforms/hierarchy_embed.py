import torch


def norm_hierarchy_embed(A, max_value):
    num_rows, num_columns = A.shape[0], max_value

    # Create column indices
    col_indices = torch.arange(num_columns, device=A.device).view(1, -1)
    col_indices = col_indices.repeat(num_rows, 1)

    # Expand A to match the shape of col_indices
    A_expanded = A.view(-1, 1).repeat(1, num_columns)

    # Create condition mask
    condition_mask = col_indices < A_expanded

    # Create divisor
    divisor = A_expanded.clone()
    divisor[A_expanded < 1.] = 1.

    # Compute the result using the condition mask
    B = torch.where(condition_mask, 1 / divisor, torch.tensor(0.0, device=A.device))
    return B


def test():
    A = torch.tensor([[1, 2, 3, 4, 5, 6]])
    B = norm_hierarchy_embed(A, 7)
    print(B)


if __name__ == "__main__":
    test()
