import openai

def generate_prompt_complete_sentences(user_input):
    return """Complete the sentence if it misses the subject.

    User: am sad
    Answer: I am sad.
    User: really enjoy my work recently.
    Answer: I really enjoy my work recently.
    User: have problem hearing you well.
    Answer: I have problem hearing you well.
    User: I am so depressed.
    Answer: I am so depressed.
    User:{}
    Answer:""".format(
            user_input.capitalize()
        )

def generate_complete_sentences(question_text):
    openai.api_key = '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'
    user_input = question_text
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=generate_prompt_complete_sentences(user_input),
        temperature=0.8,
        max_tokens = 1000,
    )
    results = response.choices[0].text
    # logger.info(response.choices[0].text)
    return results

result = generate_complete_sentences(" I am not happy")
print(result)