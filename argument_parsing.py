import argparse

def parse_args():
    parser = argparse.ArgumentParser(description= 'This is a test.')

    # Here we add an argument to the parser, specifying the expected type, a help message, etc.
    parser.add_argument('-bs', type = str, required=True, help = 'Please provide a batch size')

    return parser.parse_args()

def main():
    args = parse_args()

    # Now we can use the argument value in our program
    print(f'Batch size: {args.bs}')

if __name__ == '__main__':
    main()