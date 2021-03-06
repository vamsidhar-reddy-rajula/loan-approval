from setuptools import find_packages, setup

setup(
    name='loan_application_classifier',
    packages=find_packages(),
    version='0.1.0',
    description='Predict whether a loan will be approved or not',
    author='Vamsidhar',
    license='MIT',
    install_requires=[
        'seaborn==0.11.1',
        'matplotlib==3.4.2',
        'plotly==5.1.0',
        'statsmodels==0.12.2',
        'click==7.1.2',
        'joblib==1.0.1',
        'Flask_RESTful==0.3.9',
        'Flask==2.0.1',
        'numpy==1.21.0',
        'pandas==1.3.0',
        'python-dotenv==0.19.0',
        'scikit_learn == 0.24.2v'
    ],
)
