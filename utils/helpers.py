import time

def log(msg, end='\n'):
    print(time.strftime('[%Y-%m-%d %H:%M:%S]') + msg, end=end)