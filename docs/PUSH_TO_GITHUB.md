# Publish this scaffold to GitHub

Recommended repository name: `active-counterfactual-microscopy`

After creating an empty GitHub repository under `uskutsav-cpu` (do not initialize it with a README/license), run:

```bash
cd active-counterfactual-microscopy
git remote add origin git@github.com:uskutsav-cpu/active-counterfactual-microscopy.git
git branch -M main
git push -u origin main
```

If you use HTTPS instead of SSH:

```bash
git remote add origin https://github.com/uskutsav-cpu/active-counterfactual-microscopy.git
git branch -M main
git push -u origin main
```

Once the remote exists, the GitHub connector can also add or update files, open issues/PRs, and manage the research workflow directly.
