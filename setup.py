from setuptools import find_packages,setup
"""what this find_packages module will do 
it will scan all the folders and where there will be a __init__ file it will consider
that folder aKhuranas a package
"""
from typing import List


def get_requirements()->List[str]:
    """
This function will return the list of requirements
    """
    requirement_list:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            ##read lines
            lines=file.readlines()
            ##process each line
            for line in lines:
                requirement=line.strip()
                
                ##ignore empty lines and -e.
                if requirement and requirement  !="-e .":
                    requirement_list.append(requirement)
            
    except FileNotFoundError:
        print("requirements.txt file not found")  
        
        
    return requirement_list


setup(
    name="Agentic ai",
    version="0.0.1",
    author="Garv ",
    author_email="garvkhurana1234567@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)     








