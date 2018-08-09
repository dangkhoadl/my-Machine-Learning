

# Setting GPU on [https://www.floydhub.com/](https://www.floydhub.com/)
- Activate conda env

		cd to the local notebook dir
		source activate py36

- Install Floyd
		
		pip install -U floyd-cli

- Login 

		floyd login

## Project
- Create a project on [https://www.floydhub.com/](https://www.floydhub.com/)
- Init a project
	
		floyd init <project-name>

- Run GPU on notebook

		floyd run --env keras --mode jupyter --gpu

- Stop a project
	
		floyd stop <project-name>
