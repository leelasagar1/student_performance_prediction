# End to End project

docker build -t testdockersagar.azurecr.io/mltest:latest .


docker login testdockersagar.azurecr.io
docker push  testdockersagar.azurecr.io/mltest:latest