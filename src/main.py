import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True, help='Stock ticker symbol (e.g., AAPL)', choices=['data', 'Q'])
parser.add_argument('--prompt', required=False, help='Stock ticker symbol (e.g., AAPL)')
parser.add_argument('--stock_code', required=False, help='Financial item to visualize (e.g., "totalAssets")')
parser.add_argument('--item', required=False, help='Financial item to visualize (e.g., "totalAssets")')
parser.add_argument('--freq', required=False, choices=['A', 'Q'],
                    help='Frequency of the data: "A" for Annual, "Q" for Quarterly')

# Parse command line arguments
args = parser.parse_args()
print()

if args.mode is None:
    raise Exception("Arg -- 'mode' is required")
else:
    mode = args.mode

if mode == 'prompt':
    if args.prompt is None:
        raise Exception("Arg -- 'prompt' is required in Prompt mode")
    else:
        prompt = args.prompt
        response.generate_text_from_prompt(prompt)
elif mode == 'data':

    if args.stock_code is None or args.item is None or args.freq is None:
        raise Exception("Arg -- 'stock code' / 'item' / 'freq' are all required in data mode")
    else:
        stock_code = args.stock_code
        item = args.item
        freq = args.freq
        response.visualize_data(stock_code, item, freq)

else:
    raise Exception("Incorrect mode input -- only prompt / data mode is accepted")