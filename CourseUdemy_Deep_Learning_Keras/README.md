
# My notes for the course [Zero to Deep Learning™ with Python and Keras](https://www.udemy.com/zero-to-deep-learning/)

## Setting GPU on [https://www.floydhub.com/](https://www.floydhub.com/)
- Activate conda env

		cd to the local notebook dir
		source activate py36

- Install Floyd
		
		pip install -U floyd-cli

- Login 

		floyd login

## Dataset
- Create a dataset on [https://www.floydhub.com/datasets](https://www.floydhub.com/datasets)
- Init a dataset
	
		floyd data init <dataset-name>

- Upload dataset(After cd to datset directory)

		floyd data upload

## Project
- Create a project on [https://www.floydhub.com/projects](https://www.floydhub.com/projects)
- Init a project
	
		floyd init <dangkhoadl/project-name>

- Run CPU/GPU on notebook

		floyd run --env keras --mode jupyter --cpu --data <dangkhoadl/dataset-name>
		floyd run --env keras --mode jupyter --gpu --data <dangkhoadl/dataset-name>

- Stop a project

		floyd stop <project-name>
