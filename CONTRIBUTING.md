# Contributing To Nemo-RL

Thanks for your interest in contributing to Nemo-RL!

## Setting Up

### Development Environment

1. **Build and run the Docker container**:
```sh
docker buildx build -t nemo-rl:latest -f Dockerfile .
```

To start a shell in the container to interactively run/develop:
```sh
# Run the container with your local nemo-rl directory mounted
docker run -it --gpus all -v /path/to/nemo-rl:/nemo-rl nemo-rl:latest
```

If you are using VSCode/Cursor you can also use Dev Containers. Here's a devcontainer.json to get you started:
```jsonc
{
    "name": "rl-dev",
    "image": "nemo-rl:latest",
    "runArgs": [
        "--gpus",
        "all",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--shm-size=24g",
        "--privileged",
        "--pid=host"
	]

    // NOTE: Here is an example of how you can set up some common mounts, environment variables, and set up your shell.
    //       Feel free to adapt to your development workflow and remember to replace the user `terryk` with your username.

    //"mounts": [
    //    {"source": "/home/terryk", "target": "/home/terryk", "type": "bind"},
    //    {"source": "/home/terryk/.ssh", "target": "/root/terryk-ssh", "type": "bind"}
    //],
    //"containerEnv": {
    //    "HF_TOKEN_PATH": "/home/terryk/.cache/huggingface/token",
    //    "HF_HOME": "/home/terryk/.cache/huggingface",
    //    "HF_DATASETS_CACHE": "/home/terryk/.cache/huggingface/datasets",
    //    "WANDB_API_KEY": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    //},
    // // This (1) marks all directories safe (2) copies in ssh keys (3) sources user's bashrc file
    //"postStartCommand": "git config --global --add safe.directory '*' && cp -r /root/terryk-ssh/* /root/.ssh/ && source /home/terryk/.bashrc"
}
```

## Making Changes

### Workflow: For External Contributors (Fork Required)

#### Before You Start: Install pre-commit

Pre-commit checks (using `ruff`/`pyrefly`) will help ensure your code follows our formatting and style guidelines.

If you're an external contributor, you'll need to fork the repository:

1. **Create a fork**: Click the "Fork" button on the [GitHub repository page](https://github.com/NVIDIA-NeMo/RL) or follow this direct link: https://github.com/NVIDIA-NeMo/RL/fork

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/RL nemo-rl
   cd nemo-rl
   ```

3. **Add upstream remote** to keep your fork updated:
   ```bash
   git remote add upstream https://github.com/NVIDIA-NeMo/RL.git
   ```

4. **Install pre-commit**:
   ```bash
   # Requires `uv` to be installed
   uv run --group dev pre-commit install
   ```

5. **Keep your fork updated** before starting new work:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git push origin main
   ```

6. **Create a new branch** for your changes:
   ```bash
   git checkout main
   git switch -c your-feature-name
   ```

7. **Make your changes and commit** them:
   ```bash
   git add .
   git commit --signoff -m "Your descriptive commit message"
   ```

We require signing commits with `--signoff` (or `-s` for short). See [Signing Your Work](#signing-your-work) for details.

8. **Push to your fork**:
   ```bash
   git push origin your-feature-name
   ```

9. **Create a pull request** from your fork's branch to the main repository's `main` branch through the GitHub web interface. For example, if your GitHub username is `terrykong` and your feature branch is `your-feature-name`, the compare URL would look like: https://github.com/NVIDIA-NeMo/RL/compare/main...terrykong:RL:your-feature-name?expand=1

### Workflow: For NVIDIA Contributors (Direct Access)

If you have write access to the repository (NVIDIA contributors):

1. Clone the repository directly:
   ```bash
   git clone https://github.com/NVIDIA-NeMo/RL nemo-rl
   cd nemo-rl
   ```

2. **Install pre-commit** from the [`nemo-rl` root directory](.):
   ```bash
   # Requires `uv` to be installed
   uv run --group dev pre-commit install
   ```

3. Create a new branch for your changes:
   ```bash
   git switch -c your-feature-name
   ```

4. Make your changes and commit them:
   ```bash
   git add .
   git commit --signoff -m "Your descriptive commit message"
   ```

5. Push your branch to the repository:
   ```bash
   git push origin your-feature-name
   ```

6. Create a pull request from your branch to the `main` branch.

### Design Documentation Requirement

**Important**: All new key features (ex: enabling a new parallelization technique, enabling a new RL algorithm) must include documentation update (either a new doc or updating an existing one). This document update should:

- Explain the motivation and purpose of the feature
- Outline the technical approach and architecture
- Provide clear usage examples and instructions for users
- Document internal implementation details where appropriate

This ensures that all significant changes are well-thought-out and properly documented for future reference. Comprehensive documentation serves two critical purposes:

1. **User Adoption**: Helps users understand how to effectively use the library's features in their projects
2. **Developer Extensibility**: Enables developers to understand the internal architecture and implementation details, making it easier to modify, extend, or adapt the code for their specific use cases

Quality documentation is essential for both the usability of Nemo-RL and its ability to be customized by the community.

## Code Quality

- Follow the existing code style and conventions
- Write tests for new features
- Update documentation to reflect your changes
- Ensure all tests pass before submitting a PR
- Do not add arbitrary defaults for configs, be as explicit as possible.


## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```
