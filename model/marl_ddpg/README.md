# MARL checkpoint directory

The training scripts save actor, critic, target-network, and global-critic
checkpoints in this directory.

Checkpoint files are intentionally ignored by git because they are generated
artifacts and can be large. Keep this directory in the repository so checkpoint
save/load paths exist after cloning.
