
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_and_save_dataset(n_samples=100000, filename="tasks.csv"):
    """
    Generates a synthetic dataset of tasks and saves it to a specified CSV file.
    """
    print(f"Generating a synthetic dataset with {n_samples} samples...")
    
    # Define the building blocks for the dataset
    categories = ['Development', 'Testing', 'Documentation', 'Design', 'Marketing', 'Research', 'Bug Fix', 'Feature']
    priorities = ['Low', 'Medium', 'High', 'Critical']
    statuses = ['To Do', 'In Progress', 'Review', 'Done']
    
    # Task description templates for realism
    task_templates = {
        'Development': ['Implement user authentication system', 'Create REST API endpoints', 'Develop frontend components', 'Integrate payment gateway'],
        'Testing': ['Write unit tests for user service', 'Perform integration testing on API', 'Execute load testing scenarios'],
        'Documentation': ['Update API documentation', 'Create user manual for new features', 'Write technical specifications'],
        'Design': ['Design user interface mockups', 'Create brand identity guidelines', 'Design responsive web layouts'],
        'Marketing': ['Launch social media campaign', 'Create email marketing templates', 'Develop content marketing strategy'],
        'Research': ['Research competitor analysis', 'Investigate new technology trends', 'Analyze user behavior patterns'],
        'Bug Fix': ['Fix login authentication issue', 'Resolve database connection errors', 'Fix mobile responsiveness bug'],
        'Feature': ['Add dark mode toggle', 'Implement search functionality', 'Add export data feature', 'Create notification system']
    }
    
    data = []
    
    # Loop to create each task record
    for i in range(n_samples):
        category = random.choice(categories)
        description = random.choice(task_templates[category])
        
        # Add random variations to descriptions
        if random.random() < 0.3:
            description += f" ({random.choice(['urgent','critical','asap','important'])})"
            
        # Use weighted probabilities for priorities based on category
        priority_weights = {
            'Bug Fix': [0.1, 0.2, 0.4, 0.3], 'Development': [0.2, 0.4, 0.3, 0.1],
            'Testing': [0.3, 0.4, 0.2, 0.1], 'Documentation': [0.4, 0.4, 0.15, 0.05],
            'Design': [0.3, 0.4, 0.25, 0.05], 'Marketing': [0.25, 0.45, 0.25, 0.05],
            'Research': [0.35, 0.45, 0.15, 0.05], 'Feature': [0.2, 0.4, 0.35, 0.05]
        }
        
        # Select priority based on weights and boost it if keywords are present
        priority_idx = np.random.choice(4, p=priority_weights[category])
        if any(word in description.lower() for word in ['urgent', 'critical', 'asap', 'important']):
            priority_idx = min(3, priority_idx + 1)
        priority = priorities[priority_idx]
        
        # Generate other realistic metadata
        created_date = datetime.now() - timedelta(days=random.randint(1, 90))
        due_date = created_date + timedelta(days=random.randint(1, 30))
        effort_map = {'Low': [1, 2, 3], 'Medium': [3, 5, 8], 'High': [8, 13, 21], 'Critical': [13, 21, 34]}
        effort = random.choice(effort_map[priority])
        
        data.append({
            'task_id': f"TASK-{i+1:04d}",
            'description': description,
            'category': category,
            'priority': priority,
            'assignee': f"user_{random.randint(1, 20)}",
            'reporter': f"user_{random.randint(1, 15)}",
            'status': random.choice(statuses),
            'created_date': created_date.strftime('%Y-%m-%d'),
            'due_date': due_date.strftime('%Y-%m-%d'),
            'effort_points': effort,
            'workload_percentage': random.randint(20, 80)
        })
        
    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    print(f"✅ Success! Dataset saved to '{filename}'.")

if __name__ == "__main__":
    generate_and_save_dataset()
