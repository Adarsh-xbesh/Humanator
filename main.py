from dotenv import load_dotenv
import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

# define port number
PORT = int(os.getenv("PORT", 8000))


# Initialize FastAPI app
app = FastAPI()

# Pydantic model for input validation
class AIHumanizerInput(BaseModel):
    ai_generated_text: str


# Set up the AI Model
def setup_model():
    Mistral_API_KEY = os.getenv("Mistral_API_KEY")
    model = ChatMistralAI(api_key=Mistral_API_KEY)

# Prompt template with enhanced details
    prompt_template = """
    ### Humanizing AI-Generated Text

    You are a skilled writer tasked with rewriting AI-generated text to make it sound more human, engaging, and relatable. Additionaly, provide a score out of 100 indicating how human-like the text is based on the following criteria and also consider these criteria to generate nearly human-like text:

    1. Repetitive Phrasing: AI text often repeats phrases or sentence structures due to reliance on learned patterns
    2. Predictable Patterns: AI models can produce formulaic text that lacks human-like variability.
    3. Unnatural Language Use: AI text may include awkward or contextually inappropriate language.
    4. Lack of Contextual Understanding: AI might generate contextually irrelevant or shallow content.
    5. Overuse of Certain Words: AI tends to overuse specific words or phrases such as thrilled, revolutionized, embarking and etc, reflecting its training data.
    6. Inconsistent Tone and Style: AI text can have abrupt tone or style shifts, unlike consistent human writing.
    7. Surface-Level Content: AI often generates shallow content lacking depth and detailed analysis.
    8. Inaccuracies and Hallucinations: AI may produce factually incorrect or implausible information.
    9. Overly Formal or Stiff Language: AI text might be overly formal or lack a natural, conversational tone and feel.
    10. Lack of Personal Touch: Human writing includes personal anecdotes, emotions, and unique perspectives that AI text lacks.

    Please follow these instructions:

    **Humanizing Instructions:**

    To generate human-written text or convert AI-written text into humanized content, here are some instructions you can give to a Large Language Model (LLM):
    For generating human-written text:

    1. Write in a conversational tone: Imagine you're having a discussion with a Friend. Use everyday language and avoid jargon or overly formal tone.
    2. Add personal touches: Include personal anecdotes, emotions, or experiences to make the text more relatable and authentic.
    3. Use contractions and colloquialisms: Incorporate contractions (e.g., "don't" instead of "do not") and colloquial expressions (e.g., "break a leg") to give the text a more human feel.
    4. Vary sentence structure: Mix up short and long sentences to create a natural flow, just like human writing.
    5. Show, don't tell: Instead of stating facts, use descriptive language to paint a picture in the reader's mind.
    6. Add a little bit of related and context driven humour when you can, it must not be forced.
    7. Add a little bit of philosophy or famous quotes whenever its feasible.

    For converting AI-written text into humanized content:
    1. Rewrite in a more conversational tone: Take the AI-written text and rephrase it in a more relaxed, everyday language.
    2. Add emotional depth: Inject emotions, empathy, or personal experiences to make the text more relatable and engaging.
    3. Use more descriptive language: Replace generic terms with vivid, descriptive words to add flavor and personality to the text.
    4. Break up long sentences: Split lengthy sentences into shorter, more manageable ones to improve readability.
    5. Add idioms and colloquialisms: Incorporate common idioms and colloquial expressions to give the text a more human-like quality.
    6. Read it aloud: Read the text aloud to ensure it sounds natural and conversational.
    7. Try to touch on to user's personal experiences or general everyday life experiences.

    Additional tips:

    Use active voice: It's more engaging and easier to read than passive voice.
    Show vulnerability: Share imperfections, doubts, or fears to make the text more relatable and authentic.
    Use humor: Humor can help humanize the text and make it more enjoyable to read.
    Edit and refine: Review the text multiple times to refine the language, tone, and flow.

    **Contextual Understanding and Audience Targeting:**

    1. **Understand the Context**: Ensure the text fits well within the broader narrative and context it is a part of.
    2. **Target Audience**: Tailor the language, tone, and style to the intended audience. Consider what would resonate most with them.

    **AI-Generated Text:**
    {ai_generated_text}

    **Humanized and Accurate Text:**
    **Human-Like Score:**

    """

    # Create a Langchain prompt with the template
    prompt = PromptTemplate(
        input_variables=["ai_generated_text"],
        template=prompt_template,
    )

    chain = LLMChain(llm=model, prompt=prompt)
    return chain


# Define the API endpoint
@app.post("/humanize")
async def humanize_text(input: AIHumanizerInput):

    if len(input.ai_generated_text) == 0:
        raise HTTPException(status_code=422, detail="Please Input Some Text")

    # validating the input text
    if len(input.ai_generated_text) < 10:
        raise HTTPException(status_code=422, detail="Input text is too small")
    
    try:
        # Initialize the model
        chain = setup_model()
        # Generate humanized text
        result = chain.run(ai_generated_text=input.ai_generated_text)
        # Assuming result is a dictionary with 'humanized_text' and 'evaluation_score'
        return {
            "humanized_text": result,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Run the FastAPI server using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)