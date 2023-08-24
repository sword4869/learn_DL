```bash
pip install gradio 
```
```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    
demo.launch()   
```
The core Interface class is initialized with three required parameters:

- `fn`: the function to wrap a UI around
- `inputs`: "text", "image" or "audio"
- `outputs`: "text", "image" or "label"

```python
import gradio as gr
def generate(color, guidance_loss_scale): # label 对应参数
    pass # Implement your sketch recognition model here...

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.ColorPicker(label="color"),
        gr.Slider(label="guidance_scale", minimum=0, maximum=30)
    ],
    outputs=outputs,
    examples=[
        ["#BB2266", 3],
        ["#44CCAA", 5],  # You can provide some example inputs to get people started
    ],
)
demo.launch(debug=True)  # debug=True allows you to see errors and output in Colab
```

```python
# Components Attributes
gr.Textbox(lines=2, placeholder="Name Here...")
gr.Image(shape=(200, 200))
gr.Image(type="filepath", shape=...)
gr.ColorPicker(label="color", value="55FFAA")
gr.Slider(label="guidance_scale", minimum=0, maximum=30, value=3)

# Multiple Components
["text", "checkbox", gr.Slider(0, 100)]
```