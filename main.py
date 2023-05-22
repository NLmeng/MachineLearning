import argparse
import src.Chatbot as Chatbot

# run export OPENAI_API_KEY='..'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the chatbot.')
    parser.add_argument('--new', action='store_true', help='Create a new index.')
    args = parser.parse_args()

    chatbot = Chatbot.Chatbot("personalData")
    if args.new:
        chatbot.construct_new_index()
    chatbot.launch_interface()