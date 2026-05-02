import shutil
import sys
from pathlib import Path

def setup_subject(subject_id):
    """
    Creates the folder structure and copies the template notebooks
    for a new subject into the analysis_workspace folder.
    """
    # Define base paths (assuming the script is in the /scripts folder)
    base_dir = Path(__file__).resolve().parent.parent
    templates_dir = base_dir / "templates"
    workspace_dir = base_dir / "analysis_workspace" / subject_id

    # 1. Check if the templates folder exists
    if not templates_dir.exists():
        print(f"Error: The templates folder was not found at: {templates_dir}")
        return

    # 2. Check if the subject folder already exists to avoid overwriting data
    if workspace_dir.exists():
        print(f"Warning: The folder for {subject_id} already exists in analysis_workspace.")
        confirm = input("Do you want to add new templates to this folder? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation canceled.")
            return
    else:
        # Create the subject folder
        workspace_dir.mkdir(parents=True)
        print(f"Folder created: {workspace_dir}")

    # 3. Find all .ipynb files in the templates folder
    templates = list(templates_dir.glob("*.ipynb"))

    if not templates:
        print("No template files (.ipynb) found in the templates/ folder.")
        return

    # 4. Copy and rename the files
    for template_path in templates:
        # Define the new name: replace 'template' with the subject ID
        # Example: 01_pre_processing_template.ipynb -> 01_pre_processing_sub-01.ipynb
        new_name = template_path.name.replace("template", subject_id)
        destination = workspace_dir / new_name

        if destination.exists():
            print(f"File already exists and will not be overwritten: {new_name}")
        else:
            shutil.copy(template_path, destination)
            print(f"Copied: {template_path.name} -> {new_name}")

    print(f"\n✅ Environment successfully configured for {subject_id}!")

if __name__ == "__main__":
    # Check if the subject ID was passed via command line
    # Usage example: python scripts/setup_subject.py sub-01
    if len(sys.argv) < 2:
        print("Correct usage: python scripts/setup_subject.py <SUBJECT_ID>")
        print("Example: python scripts/setup_subject.py sub-05")
    else:
        subject_id_input = sys.argv[1]
        setup_subject(subject_id_input)