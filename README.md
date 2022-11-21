# FootAI

A repo for doing player-based Football match prediction. 

This project contains everything that is needed to do predict the outcome of any football game between 
teams that FIFA has player data for, from games between international teams to a league played in a country.

The project currently focuses on predicting the difference in goals between the home and the away teams. 
However, solutions are also included for predicting the total end result.

The SHAP explaining is still in developmental phase and is not correctly functional (`shap_explaining.py`)

Check out this <a href="[https://www.vespertool.com/](https://medium.com/@rvdb7345/world-cup-2022-prediction-paul-the-octopus-vs-machine-learning-1c825038436d
)">Medium article</a> to get an introduction to the project. 


<div align="center">
  
[![GitHub release](https://img.shields.io/github/release/rvdb7345/FootAI?include_prereleases=&sort=semver&color=blue)](https://github.com/rvdb7345/FootAI/releases/)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license)
[![issues - FootAI](https://img.shields.io/github/issues/rvdb7345/FootAI)](https://github.com/rvdb7345/FootAI/issues)
  
[![rvdb7345 - FootAI](https://img.shields.io/static/v1?label=rvdb7345&message=FootAI&color=blue&logo=github)](https://github.com/rvdb7345/FootAI "Go to GitHub repo")
[![stars - FootAI](https://img.shields.io/github/stars/rvdb7345/FootAI?style=social)](https://github.com/rvdb7345/FootAI)
[![forks - FootAI](https://img.shields.io/github/forks/rvdb7345/FootAI?style=social)](https://github.com/rvdb7345/FootAI)
 
</div>

## What part of the prediction process is covered here?
Thought the scripts in this repo, the following functionalities are implemented in the corresponding files:
- Scraping historic fixture data (`collect_fixtures.py`)
- Aggregating historic fixture data (`create_fixture_rdb.py`)
- Aligning team names in fixtures data with those in the FIFA player dataset (`match_club_names_fifa_fixutres.py`)
- Composing optimal teams and engineering features based on player FIFA attributes (`feature_engineer.py`)
- Training and evaluating a light gradient boosting model (`train_evaluate_lgbm.py`)
- Training and evaluating a deep learning model (`train_evaluate_dnn.py`)
- Predicting the outcomes of new matches with saved models (`predict_new_match_lgbm.py`/`predict_new_match_dnn.py`)
- Predicting an overall ranking for a set of teams (`get_league_ranking.py`)

