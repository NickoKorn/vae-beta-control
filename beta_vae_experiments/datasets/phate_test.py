import phate

if __name__ == "__main__":

    tree, branches = phate.tree.gen_dla(n_dim = 200, n_branch = 10, branch_length = 300,
                                        rand_multiplier = 2, seed=37, sigma = 5)
    print(tree.shape, branches.shape)
    phate_operator = phate.PHATE(n_jobs=-2)
    tree_phate = phate_operator.fit_transform(tree)
