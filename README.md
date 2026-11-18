# IPL Match Winner Prediction — Decision Tree

A compact machine-learning project that explores predicting an IPL match winner from a small set of match-context features.

## Inputs documented
- Team 1
- Team 2
- Venue

## Model
- Decision Tree Classifier
- Reported accuracy: approximately 50%

This is an experimental baseline with limited input features. It should not be treated as a reliable match forecast. Toss information, team form, player availability, historical team/venue performance, and time-aware validation would be useful next steps.

## Project focus
- Preparing categorical match features.
- Training a tree-based classifier.
- Evaluating a baseline prediction task.

## Next improvements
1. Add toss winner and toss decision.
2. Add historical team and player performance features.
3. Use time-aware validation to avoid leakage across seasons.
4. Compare against simple majority and team-strength baselines.
5. Report class-wise precision, recall, and a confusion matrix.

## Author
[Aun Abbas](https://github.com/aunabbas007)
