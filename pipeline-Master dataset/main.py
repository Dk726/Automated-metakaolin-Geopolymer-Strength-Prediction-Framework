import os
import sys
import time
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

class PipelineRunner:
    def __init__(self):
        self.start_time = None
        self.log_file = None
        self.scripts = [
            ("Global_properties_extraction.py", self.validate_step1),
            ("Sand_local_properties.py", self.validate_step2),
            ("Unreacted_local_properties.py", self.validate_step3),
            ("Porosity_local_properties.py", self.validate_step4),
            ("Filter_local_properties.py", self.validate_step5),
            ("Master_table_create.py", self.validate_step6)
        ]
        self.setup_directories()
        
    def setup_directories(self):
        """Create all necessary directories if they don't exist"""
        directories = [
            "data/segmented_images",
            "data/global",
            "data/master",
            "data/local/sand/data",
            "data/local/sand/images",
            "data/local/unreacted/data",
            "data/local/unreacted/images",
            "data/local/porosity/data",
            "data/local/porosity/images",
            "data/filtered_Local/sand",
            "data/filtered_Local/porosity",
            "data/filtered_Local/unreacted"
        ]
        
        print("Creating directory structure...")
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"{directory}")
    
    def log_message(self, message, level="INFO"):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        print(log_entry)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + "\n")
    
    def run_script(self, script_name, step_number, total_steps):
        """Run a single Python script"""
        self.log_message(f"Step {step_number}/{total_steps}: Running {script_name}")
        
        try:
            # Check if script exists
            if not os.path.exists(script_name):
                self.log_message(f"Script not found: {script_name}", "ERROR")
                return False
            
            # Run the script
            process = subprocess.Popen(
                [sys.executable, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Capture output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(f"  {output.strip()}")
            
            # Check for errors
            return_code = process.poll()
            if return_code != 0:
                error_output = process.stderr.read()
                self.log_message(f"{script_name} failed with error:", "ERROR")
                self.log_message(f"{error_output}", "ERROR")
                return False
            
            self.log_message(f"{script_name} completed successfully")
            return True
            
        except Exception as e:
            self.log_message(f"Exception running {script_name}: {str(e)}", "ERROR")
            traceback.print_exc()
            return False
    
    def validate_step1(self):
        """Validate step 1 output"""
        output_file = "data/global/global_properties.xlsx"
        if os.path.exists(output_file):
            self.log_message(f"Step 1 validation passed: {output_file} created")
            return True
        else:
            self.log_message(f"Step 1 validation failed: {output_file} not found", "ERROR")
            return False
    
    def validate_step2(self):
        """Validate step 2 output"""
        data_dir = "data/local/sand/data"
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
            if files:
                self.log_message(f"Step 2 validation passed: {len(files)} sand property files created")
                return True
        self.log_message("Step 2 validation failed: No sand property files found", "ERROR")
        return False
    
    def validate_step3(self):
        """Validate step 3 output"""
        data_dir = "data/local/unreacted/data"
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
            if files:
                self.log_message(f"Step 3 validation passed: {len(files)} unreacted property files created")
                return True
        self.log_message("Step 3 validation failed: No unreacted property files found", "ERROR")
        return False
    
    def validate_step4(self):
        """Validate step 4 output"""
        data_dir = "data/local/porosity/data"
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
            if files:
                self.log_message(f"Step 4 validation passed: {len(files)} porosity property files created")
                return True
        self.log_message("Step 4 validation failed: No porosity property files found", "ERROR")
        return False
    
    def validate_step5(self):
        """Validate step 5 output"""
        filtered_dirs = [
            "data/filtered_Local/sand",
            "data/filtered_Local/porosity",
            "data/filtered_Local/unreacted"
        ]
        
        for dir_path in filtered_dirs:
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.endswith('.xlsx')]
                if files:
                    self.log_message(f"Step 5 validation passed: {len(files)} filtered files in {os.path.basename(dir_path)}")
                else:
                    self.log_message(f"Step 5 warning: No filtered files in {os.path.basename(dir_path)}", "WARNING")
        return True
    
    def validate_step6(self):
        """Validate step 6 output"""
        output_file = "data/master/master.xlsx"
        if os.path.exists(output_file):
            self.log_message(f"Step 6 validation passed: Master dataset created at {output_file}")
            return True
        else:
            self.log_message(f"Step 6 validation failed: {output_file} not found", "ERROR")
            return False
    
    def check_input_images(self):
        """Check if input images exist"""
        input_folder = "data/segmented_images"
        if not os.path.exists(input_folder):
            self.log_message(f"Input folder not found: {input_folder}", "ERROR")
            return False
        
        images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        if not images:
            self.log_message("No images found in segmented_images folder", "ERROR")
            self.log_message("Supported formats: .png, .jpg, .jpeg, .tif, .tiff", "INFO")
            return False
        
        self.log_message(f"Found {len(images)} image(s) in segmented_images folder")
        return True
    
    def run_pipeline(self):
        """Main pipeline execution method"""
        self.start_time = datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_file = f"pipeline_log_{timestamp}.txt"
        
        print("\n" + "="*60)
        print("IMAGE ANALYSIS PIPELINE - STARTING")
        print("="*60)
        
        # Log start
        self.log_message(f"Pipeline started at {self.start_time}")
        self.log_message(f"Log file: {self.log_file}")
        
        # Check input images
        if not self.check_input_images():
            return False
        
        # Run each script in sequence
        total_steps = len(self.scripts)
        for i, (script, validator) in enumerate(self.scripts, 1):
            success = self.run_script(script, i, total_steps)
            
            if not success:
                self.log_message(f"Pipeline failed at step {i}: {script}", "ERROR")
                self.log_message("Stopping pipeline...", "ERROR")
                return False
            
            # Validate step output
            if not validator():
                self.log_message(f"Validation failed for step {i}: {script}", "ERROR")
                return False
            
            # Brief pause between steps
            if i < total_steps:
                time.sleep(1)
        
        # Calculate total time
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        self.log_message(f"Pipeline completed at {end_time}")
        self.log_message(f"Total duration: {duration}")
        
        # Show output summary
        self.show_summary()
        
        return True
    
    def show_summary(self):
        """Show summary of generated files"""
        print("\n" + "="*60)
        print("OUTPUT SUMMARY")
        print("="*60)
        
        # Global properties
        global_file = "data/global/global_properties.xlsx"
        if os.path.exists(global_file):
            import pandas as pd
            try:
                df = pd.read_excel(global_file)
                print(f"Global properties: {len(df)} images analyzed")
            except:
                print(f"Global properties file created")
        
        # Local properties count
        for phase in ['sand', 'unreacted', 'porosity']:
            data_dir = f"data/local/{phase}/data"
            if os.path.exists(data_dir):
                files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
                print(f"{phase.capitalize()} local: {len(files)} property files")
        
        # Filtered properties count
        for phase in ['sand', 'unreacted', 'porosity']:
            data_dir = f"data/filtered_Local/{phase}"
            if os.path.exists(data_dir):
                files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
                print(f"{phase.capitalize()} filtered: {len(files)} files")
        
        # Master dataset
        master_file = "data/master/master.xlsx"
        if os.path.exists(master_file):
            import pandas as pd
            try:
                df = pd.read_excel(master_file)
                print(f"Master dataset: {len(df)} rows, {len(df.columns)} columns")
                print(f"  File saved: {os.path.abspath(master_file)}")
            except:
                print(f"Master dataset created")
        
        print("="*60)


def main():
    """Main function"""
    print("Image Analysis Pipeline")
    print("="*60)
    print("This will run all analysis scripts in sequence:")
    print("1. Global properties extraction")
    print("2. Sand local properties")
    print("3. Unreacted local properties")
    print("4. Porosity local properties")
    print("5. Filter local properties")
    print("6. Master table creation")
    print("="*60)
    
    # Ask for confirmation
    response = input("\nDo you want to continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Pipeline cancelled.")
        return
    
    # Run pipeline
    pipeline = PipelineRunner()
    success = pipeline.run_pipeline()
    
    if success:
        print("\nAll tasks completed successfully!")
        print("Check the log file for detailed output.")
    else:
        print("\nPipeline failed. Check the log file for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()