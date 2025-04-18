import os

def create_directory_structure():
    directories = [
        'airflow/dags',
        'data/raw',
        'data/processed',
        'data/metadata',
        'training/configs',
        'training/scripts',
        'training/parallel_launcher',
        'evaluation/metrics',
        'tensorboard_logs',
        'notebooks',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create an empty __init__.py file in each Python package directory
        if not directory.startswith(('data/', 'tensorboard_logs/', 'notebooks/', 'results/')):
            init_file = os.path.join(directory, '__init__.py')
            open(init_file, 'a').close()

if __name__ == "__main__":
    create_directory_structure()
    print("Project directory structure created successfully!") 