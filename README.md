# Solving NLP Problems through Human-System Collaboration: A Discussion-based Approach

If you use any part of this work, make sure you include the following citation:
```
@inproceedings{Kaneko:eacl:2024,
    title={Solving NLP Problems through Human-System Collaboration: A Discussion-based Approach},
    author={Kaneko, Masahiro and Neubig, Graham and Okazaki, Naoaki},
    booktitle={EACL},
    year={2024}
}
```


## Inference using the proposed method.

```
python nli.py --input $nli_dataset_path --output $output_path --learning [zero-shot, few-shot, few-shot_humans_discussion] --task [anli1, anli2, anli3, snli] --example examples.json
```

## Dataset

discussion1.csv and discussion2.csv are validation datasets.
