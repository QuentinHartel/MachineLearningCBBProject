# MachineLearningCBBProject
### Author: Quentin Hartel

## Project Description 

## Data Pipeline
The data is retrevied from https://collegebasketballdata.com though an API. I grab game, team, and betting line data combinding them for usage with predicting the final score difference of a college basketball game. The pipeline makes sure that each game observation only includes team data from prior games of that season so data leakage is avoided. This pipeline also deals with a major piece of missing data being D1 vs non-D1 teams as there is no data on non-D1 teams so for now I am just avoiding those games with the exeption of rest time and a few other factors. The reason for this is major teams dont really play those games and when smaller teams do they usually act as major outliers compared to what the team can actually do. Then the pipeline normalizes the data as that will increase the accuracy of machine learning models by making it easier to add weiths and measure features error contributions. Lastly the pipeline will make a csv for each season which is a design decision based on how I must do my training test splits to avoid data leakage because each seasons observations are not independant within seasons.

## Feature Selection