ai-indent
==============================

## Version 1

Version 1 of the model was the first version that I tried out in VSCode using the model to predict indentation of the next line, making use of the ontypeformatting call that's part of the LSP. This "worked", and was right a good percentage of the time, and was responsive (inference time took about 3ms, even running on CPU).

Using this, two areas of potential improvement that we identified were:
- No context after the cursor position. Context which could turn a blind guess into a certainty
- Context gained after typing the next line could be used to re-predict the line with much better accuracy

## Version 2

#### Planned improvements/work
1. Predict next line, and re-predict current line
  - Only re-predict current line if the line has been modified (e.g. formatters which take into account git history)
2. Give the model context before and after the cursor
3. Remove lines that are just whitespace before performing inference
4. Perform simple de-duplication on training data

## Version 3

#### Planned improvements/work

1. Take into account the type and amount of indexation set in the IDE 
   - This should also have a way of saying 'undefined indentation settings'
   - It should also be able to identify files that have different styles, and so the IDE should be ignored
2. To produce training data of mixed indentation styles (e.g. tabs vs spaces), I plan to make use of some Ada formatter...
4. Perform near de-duplication on training data: https://chenghaomou.github.io/posts/20230220150602


## Version X
- Investigate using context from other files in the project. Useful for new/small files
- Figure out what to do with non-standard with UTF-8 characters 😭


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
