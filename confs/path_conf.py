import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

project_root = project_root.replace('\\', '/')
print('project_root:', project_root)

system_experience_dir = project_root + '/experiences/'
system_log_dir = project_root + '/logs/'
system_data_dir = project_root + '/data/'
system_model_dir = project_root + '/models/'
system_rl_model_dir = project_root + '/models/rl/'
system_ex_model_dir = project_root + '/models/ex/'


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

ensure_directory_exists(system_data_dir)
ensure_directory_exists(system_log_dir)
ensure_directory_exists(system_experience_dir)
ensure_directory_exists(system_model_dir)
ensure_directory_exists(system_rl_model_dir)
ensure_directory_exists(system_ex_model_dir)

if __name__ == '__main__':
    print(system_data_dir)

