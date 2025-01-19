# Creative Commons License annotation of CommonCrawl

## Running


## To do

- CC-MAIN-2024-51
- CC-MAIN-2023-06
- CC-MAIN-2020-05
- CC-MAIN-2021-04
- CC-MAIN-2022-05
- CC-MAIN-2019-30

## Commands

Main:

```bash
sbatch launch.slurm CC-MAIN-2024-51
```

Upload:

```bash
# Wait 15 minutes before starting and then upload folder every 30 seconds
sbatch upload_all.slurm -d /dodrio/scratch/projects/2024_107/gpt-nl-copyright/output/CC-MAIN-2024-51 -w 15 -e 30
```


---
license:
- odc-by
multilinguality:
- multilingual
language:
- af
- de
- en
- es
- fr
- fy
- it
- nl
tags:
- common crawl
- creative commons
task_categories:
- text-generation
task_ids:
- language-modeling