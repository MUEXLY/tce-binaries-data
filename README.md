# tce-binaries-data

This repo has data with binary configurations for the Ta-W, Ta-V, and Cr-V systems

## Setup instructions

This repository uses `uv` for environment management. See installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

After installing `uv`:

```bash
git clone git@github.com:MUEXLY/tce-binaries-data.git
cd tce-binaries-data
uv sync
```

## Configurations

Configurations are generated using `generate-configurations.py`. You don't need to run this, but it's probably nice to look at it and see what we're doing.

## Your instructions

For your intended system, you'll see a directory structure that looks like:

```
configurations/
    TaW/
        1/
            configuration.xyz
        2/
            configuration.xyz
```

and so on. For each of these configurations, you need to compute the fully relaxed potential energy using LAMMPS.

You can do this a variety of ways, but what we need is to fully relax each sample (including relaxing the simulation cell), and to put that relaxed energy in a text file, as such:

```
configurations/
    TaW/
        1/
            configuration.xyz
            energy.txt
        2/
            configuration.xyz
            energy.txt
```

This part is kind of purposefully vague; you should explore the LAMMPS documentation to figure out how to do this in your own preferred way. LAMMPS has an example [here](https://github.com/lammps/lammps/blob/develop/examples/min/in.min.box). I would personally do this by hooking LAMMPS into ASE, using a [FrechetCellFilter](https://docs.ase-lib.org/ase/filters.html#ase.filters.FrechetCellFilter) to relax the sample. But, this can be a bit of a pain to set up.
