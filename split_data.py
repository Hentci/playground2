import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(images_dir, response_file, output_base_dir, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2):
    """
    Split data into train, valid, and test directories according to specified ratios
    
    Args:
        images_dir: Directory containing images
        response_file: CSV file containing relevance values
        output_base_dir: Base output directory
        train_ratio: Training set ratio, default 0.7
        valid_ratio: Validation set ratio, default 0.1
        test_ratio: Test set ratio, default 0.2
    """
    # Ensure the sum of ratios equals 1
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-10, "The sum of ratios must be 1"
    
    # Create output directory structure
    train_dir = os.path.join(output_base_dir, 'train')
    valid_dir = os.path.join(output_base_dir, 'valid')
    test_dir = os.path.join(output_base_dir, 'test')
    
    for directory in [train_dir, valid_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Load response data
    response_df = pd.read_csv(response_file)
    
    # Get list of valid image IDs
    valid_ids = []
    for image_id in response_df['id']:
        image_path = os.path.join(images_dir, f"{image_id}.png")
        if os.path.exists(image_path):
            valid_ids.append(image_id)
    
    print(f"Number of valid images found: {len(valid_ids)}")
    
    # Filter response dataframe to keep only valid IDs
    filtered_df = response_df[response_df['id'].isin(valid_ids)]
    
    # First split data into training set and temporary set
    train_ids, temp_ids = train_test_split(
        filtered_df['id'].values, 
        train_size=train_ratio,
        random_state=422
    )
    
    # Then split temporary set into validation and test sets
    valid_ratio_adjusted = valid_ratio / (valid_ratio + test_ratio)
    valid_ids, test_ids = train_test_split(
        temp_ids,
        train_size=valid_ratio_adjusted,
        random_state=422
    )
    
    print(f"Training set size: {len(train_ids)} ({len(train_ids)/len(valid_ids):.1%})")
    print(f"Validation set size: {len(valid_ids)} ({len(valid_ids)/len(valid_ids):.1%})")
    print(f"Test set size: {len(test_ids)} ({len(test_ids)/len(valid_ids):.1%})")
    
    # Create set mapping
    id_to_set = {}
    for id_val in train_ids:
        id_to_set[id_val] = 'train'
    for id_val in valid_ids:
        id_to_set[id_val] = 'valid'
    for id_val in test_ids:
        id_to_set[id_val] = 'test'
    
    # Create CSV files for each set
    train_df = filtered_df[filtered_df['id'].isin(train_ids)]
    valid_df = filtered_df[filtered_df['id'].isin(valid_ids)]
    test_df = filtered_df[filtered_df['id'].isin(test_ids)]
    
    train_df.to_csv(os.path.join(train_dir, 'responses.csv'), index=False)
    valid_df.to_csv(os.path.join(valid_dir, 'responses.csv'), index=False)
    test_df.to_csv(os.path.join(test_dir, 'responses.csv'), index=False)
    
    # Copy images to appropriate directories
    copied_count = 0
    for image_id, dataset in id_to_set.items():
        src_path = os.path.join(images_dir, f"{image_id}.png")
        if os.path.exists(src_path):
            if dataset == 'train':
                dst_path = os.path.join(train_dir, f"{image_id}.png")
            elif dataset == 'valid':
                dst_path = os.path.join(valid_dir, f"{image_id}.png")
            else:  # test
                dst_path = os.path.join(test_dir, f"{image_id}.png")
            
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            
            if copied_count % 1000 == 0:
                print(f"Copied {copied_count} files...")
    
    print(f"\nData splitting completed! Total {copied_count} files copied.")

if __name__ == "__main__":
    # Set paths
    images_dir = 'correlation_assignment/images'  # Directory containing images
    response_file = 'correlation_assignment/responses.csv'  # Response CSV file
    output_base_dir = 'data'  # Base output directory
    
    # Execute splitting
    split_data(
        images_dir=images_dir,
        response_file=response_file,
        output_base_dir=output_base_dir,
        train_ratio=0.7,
        valid_ratio=0.1,
        test_ratio=0.2
    )