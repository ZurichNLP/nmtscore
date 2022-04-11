
# Human judgments from the WebNLG+ Challenge 2020

## Data sources
- https://github.com/WebNLG/challenge-2020
- https://gitlab.com/shimorina/webnlg-dataset/-/blob/master/release_v3.0/ru/test/rdf-to-text-generation-test-data-with-refs-ru.xml

## License
CC BY-NC-SA 4.0

## Changes made to the original data
- Reformatted the data as JSON lines compatible to [SacreRouge](https://github.com/danieldeutsch/sacrerouge)
- Added the human metric "overall_adequacy" by averaging correctness, data coverage and relevance
- Added the human metric "overall_fluency" by averaging fluency and text structure

## Citation
```bibtex
@inproceedings{webnlg2020report,
    title = "The 2020 Bilingual, Bi-Directional {W}eb{NLG}+ Shared Task: Overview and Evaluation Results ({W}eb{NLG}+ 2020)",
    author = "Castro Ferreira, Thiago  and
      Gardent, Claire  and
      Ilinykh, Nikolai  and
      van der Lee, Chris  and
      Mille, Simon  and
      Moussallem, Diego  and
      Shimorina, Anastasia",
    booktitle = "Proceedings of the 3rd International Workshop on Natural Language Generation from the Semantic Web (WebNLG+)",
    month = "12",
    year = "2020",
    address = "Dublin, Ireland (Virtual)",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.webnlg-1.7",
    pages = "55--76"
}
```