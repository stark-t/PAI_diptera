from tqdm import tqdm
import numpy as np

def is_convertible_to_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_console_output(log_console_path='path'):
    times = []
    train_loss = []
    val_loss = []

    d=1

    # Open the file for reading
    with open(log_console_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Use re.match to check if the line matches the pattern
            if 'Epoch' in line and '100%' in line and not 'Validation' in line:
                # get times
                line_parts = line.split(' ')
                time_parts = [f for f in line_parts if '<' in f][0]
                time_str = time_parts.split('<')[0]
                time_str = time_str.split('[')[-1]
                try:
                    minutes, seconds = map(int, time_str.split(':'))
                    total_seconds = (minutes * 60) + seconds
                    times.append(float(total_seconds))
                except Exception as e:
                    print(f"Error encountered when processing {time_str}: {e}")


            if 'Epoch' in line and 'train_loss' in line and 'val_loss' in line:
                # get losses
                line_parts = line.split(',')
                train_loss_parts = [f for f in line_parts if 'train_loss' in f][0]
                train_loss_str = train_loss_parts.split('train_loss=')[1]
                train_loss_str = train_loss_str.split(']\n')[0]
                if is_convertible_to_float(train_loss_str):
                    train_loss.append(float(train_loss_str))
                val_loss_parts = [f for f in line_parts if 'val_loss' in f][0]
                val_loss_str = val_loss_parts.split('val_loss=')[1]
                if is_convertible_to_float(val_loss_str):
                    val_loss.append(float(val_loss_str))

    median_iterationpersecond = np.median(times)

    return median_iterationpersecond, train_loss, val_loss

if __name__ == "__main__":
    get_console_output(log_console_path='path')
