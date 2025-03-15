from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
import httpx
import os
from llama_index.llms.openai import OpenAI
import openai
import contextlib
import io
import traceback
import time
import tiktoken
from tqdm import tqdm
import threading

base_model = 'o3-mini-2025-01-31'

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)

def polish_query(context,query):
    '''
    TODO
    This function is used to polish the query problem based on the previous context.
    '''
    prompt = f"**Context**:{context} **Query**:{query}"
    
    completion = client.chat.completions.create(
        model=base_model, 
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You need to polish the query based on the previous context."},
            {"role": "user", "content": f"{prompt}"}
        ]
    )
    return completion.choices[0].message.content.strip()


def google_retriever(query,retrieve_websites = 5,query_websites = 1):
    '''
    query[str]: 
    ''' 
    google_api_key=os.environ.get("GOOGLE_API_KEY", "")
    google_cx=os.environ.get("GOOGLE_CX", "")
    
    base_url = 'https://www.googleapis.com/customsearch/v1'
    params = {
            'key': google_api_key,
            'cx': google_cx,
            'q': query,
            'num':retrieve_websites # The number of websites you want.
    }
    #TODO: There might some connection error when we call httpx.get. Simply recall will resolve this error.
    response = httpx.get(base_url, params=params,timeout=5)
    response.raise_for_status()
    data = response.json()
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY", "") 
        
    urls = []
    for item in data.get('items', []):
        urls.append(item.get('link', ''))
    for i in range(retrieve_websites):
        try:
            document = SimpleWebPageReader(html_to_text=True).load_data(
            [urls[i]]
            )
            index = SummaryIndex.from_documents(document)
            llm = OpenAI(model=base_model)
            query_engine = index.as_query_engine(llm=llm)
            response = query_engine.query(query + ' Use only one paragraph to summarize your answer.')
            return response
        except:
            continue

def execute_and_capture(code: str, client, max_attempts: int = 3) -> tuple:
    """
    Execute the given Python code and capture its output and errors.
    If an error occurs, attempt to debug it using the AI model, retrying up to max_attempts times.
    """
    
    for attempt in range(max_attempts):
        stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
        def run_code():
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(code, {})

        try:
            thread = threading.Thread(target=run_code)
            thread.start()
            thread.join(timeout=30)  # Wait up to 30 seconds for the code to finish
            if thread.is_alive():
                raise TimeoutError("Code execution exceeded 30 seconds")
            if stdout_capture.getvalue().strip():  # If there's meaningful output, return it
                return code, stdout_capture.getvalue(), stderr_capture.getvalue()
            # If no output, ask AI to modify the code to include print statements
            debug_prompt = (
                f"The following Python code ran without errors but produced no output:\n{code}\n"
                "Please modify it to include meaningful print statements so that it generates useful output of the final result of this code. "
                "Only return the modified Python code."
            )
        except Exception as exc:
            if isinstance(exc, TimeoutError):
                error_output = "Execution timed out: Your Python code took too long to complete. \
                    Please review the algorithm for inefficiencies or potential bugs and consider implementing a more efficient solution."
            else:
                traceback.print_exc(file=stderr_capture)
                error_output = stderr_capture.getvalue()
            debug_prompt = (
                f"Below is a Python code snippet:\n{code}\n"
                f"The error message is:\n{error_output}\n"
                "Please debug this code and ensure it runs correctly. If it does not produce any output, modify it to include print statements so that the final result is clearly displayed. Make sure not to add any code that could terminate the process, such as using 'sys.exit(1)' or similar functions. Only return the modified Python code."
            )

            
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": debug_prompt}],
            model=base_model,
        )
        
        code = response.choices[0].message.content.strip()
    raise Exception(f"Python Debug Fail. More than {max_attempts} trails.")

def python_interpreter(code: str) -> str:
    """
    Executes the given Python code, captures its output and errors, and returns 
    an analysis report explaining the code's functionality and how the output reflects 
    the code's intended behavior.
    """

    code,output, error_output = execute_and_capture(code,client,5)
    # Prepare prompt for analysis (include code and its output)
    # analysis_prompt = (
    #     f"Below is a Python code:\n```python\n{code}\n```\n"
    #     f"The output of this code is:\n```text\n{output}{error_output}\n```\n"
    #     "Summarize the output of the Python code below in a concise and clear manner. Try to use only one sentence."
    # )
    # # Call OpenAI API for analysis (using model o3-mini-2025-01-25)
    # response = client.chat.completions.create(
    # messages=[
    #     {
    #         "role": "user",
    #         "content": f"{analysis_prompt}",
    #     }
    # ],
    # model=base_model,
    # )
    
    # response = response.choices[0].message.content.strip()

    # Format the returned report with sections
    if output:
        return code,output
    else:
        return code,error_output


if __name__ == "__main__":
    # context = ''
    # query  = "What is love? "
    # query = "What does non-commutative mean in the context of ring theory?"
    # response = google_retriever(query)
    # # # with open('./retriever_response.txt', 'w') as file:
    # # #    file.write(str(response))
    # # # # The text shows in the terminal might be incomplete. Please refer to retriever_response.txt.
    # print(response)
#     query = """
# def trap(height):
#     n = len(height)
#     if n <= 2: return 0
#     left, right = 0, n - 1
#     left_max, right_max = height[left], height[right]
#     water = 0
#     while left < right:
#         if height[left] < height[right]:
#             if height[left] >= left_max:
#                 left_max = height[left]
#             else:
#                 water += left_max - height[left]
#             left += 1
#         else:
#             if height[right] >= right_max:
#                 right_max = height[right]
#             else:
#                 water += right_max - height[right]
#             right -= 1
#     return water

# height = [0,1,0,2,1,0,1,3,2,1,2,1]
# result = trap(height)
# print(f"Total water trapped: {result}")
# """
    query = "2022/1347"
    response = python_interpreter(query)
    print(response)
