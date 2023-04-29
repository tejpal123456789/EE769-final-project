from setuptools import find_packages,setup
from typing import List

def get_requirements(filepath:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    with open(filepath) as file:
        requirements=file.readlines()
    
        #print(requirements)
        requirements=[req.replace('/n','') for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements
    


setup(
    name='EE769',
    version='0.0.1',
    description ='Final year project for the course EE769: Introduction to Machine Learning',
    author =' thetajas@gmail.com',
    packages=find_packages(),
    # What if reqyire many packages 
    # Write a function and then send it there 

    install_requires=get_requirements('requirements.txt')

)

# -e. is already in our requiremenst.txt and we are opening that file in setup.py
# we need to remove it from here because when we run setup.py it will build our packages with 
# packages in requirements apart from -e.