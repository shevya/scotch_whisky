# scotch_whisky

## Source data
Data is downloaded from http://adn.biol.umontreal.ca/~numericalecology/data/scotch.html

## Decription
Original data set contains 68 columns of characters or flavors, including body, sweetness, smoky...etc. Besides those features, there are some columns of region name, district, distillery score and scotch score.

Additionally to the original data set, I added two more columns: longitude and latitude.

## Goal
The goal of this project is to build a content-based recommendation system for choosing whiskies. Recommendation based on the similarity between whiskies.

## TODO
Next steps:
1. create a model to predict cluster, it will allow the user to make recommendations. 
2. create a model to predict dist value, it will allow the user to add new examples of whiskies. 
Use of selected algorithms for testing: knn, decision trees, gradient boosting
3. Then create an application (e.g. form) to allow users easier addition of new records.
