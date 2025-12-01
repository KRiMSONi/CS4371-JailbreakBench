"""Load specific behaviors from the JailbreakBench dataset for evaluation."""

import sys
from pathlib import Path

# Add the jailbreakbench package to the path
jbb_path = Path(__file__).parent.parent.parent / "jailbreakbench" / "src"
if jbb_path.exists():
    sys.path.insert(0, str(jbb_path))

import jailbreakbench as jbb


def load_behaviors(categories=None, max_per_category=None):
    """
    Load behaviors from the JailbreakBench dataset.
    
    Args:
        categories: List of categories to filter by (e.g., ["Malware", "Hacking"])
                   If None, loads from all categories
        max_per_category: Maximum number of behaviors to load per category
                         If None, loads all behaviors in each category
    
    Returns:
        dict: Attack prompts organized by source/category
    """
    dataset = jbb.read_dataset(split="harmful")
    df = dataset.as_dataframe()
    
    attack_prompts = {}
    
    if categories:
        # Filter by specific categories
        for category in categories:
            category_df = df[df['Category'].str.contains(category, case=False, na=False)]
            if not category_df.empty:
                if max_per_category is not None:
                    prompts = category_df['Goal'].head(max_per_category).tolist()
                else:
                    prompts = category_df['Goal'].tolist()
                attack_prompts[category] = prompts
    else:
        # Group by source
        for source in df['Source'].unique():
            source_df = df[df['Source'] == source]
            if max_per_category is not None:
                prompts = source_df['Goal'].head(max_per_category).tolist()
            else:
                prompts = source_df['Goal'].tolist()
            attack_prompts[source] = prompts
    
    return attack_prompts


def load_benign_behaviors(max_count=10):
    """
    Load benign behaviors from the JailbreakBench dataset.
    
    Args:
        max_count: Maximum number of benign prompts to check for topic
        categories: what categories to examine (i.e. "Malware")
    
    Returns:
        dict: Benign prompts organized by source/category
    """

    dataset = jbb.read_dataset(split="benign")
    df = dataset.as_dataframe()
    
    benign_prompts = {}
    
    if categories:
        # Filter by specific categories
        for category in categories:
            category_df = df[df['Category'].str.contains(category, case=False, na=False)]
            if not category_df.empty:
                if max_count is not None:
                    prompts = category_df['Goal'].head(max_count).tolist()
                else:
                    prompts = category_df['Goal'].tolist()
                benign_prompts[category] = prompts
    else:
        # Group by source
        for source in df['Source'].unique():
            source_df = df[df['Source'] == source]
            if max_count is not None:
                prompts = source_df['Goal'].head(max_count).tolist()
            else:
                prompts = source_df['Goal'].tolist()
            benign_prompts[source] = prompts
    
    return benign_prompts


# Load behaviors from the dataset
# You can customize these categories to focus on specific cyber security threats
CYBER_CATEGORIES = ["Malware", "Privacy"]

ATTACK_PROMPTS = load_behaviors(categories=CYBER_CATEGORIES, max_per_category=None)
BENIGN_PROMPTS = load_benign_behaviors(categories=CYBER_CATEGORIES,max_count=None) # max_count is how many of the data set to search through for relevant prompts
