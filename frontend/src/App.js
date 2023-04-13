import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [prompt, setPrompt] = useState('');
  const [generatedText, setGeneratedText] = useState('');

  const handleChange = (event) => {
    setPrompt(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
  
    try {
      const response = await axios.post('http://localhost:5000/generate', { prompt }, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
      setGeneratedText(response.data);
    } catch (error) {
      console.error('Error generating text:', error);
    }
  };

  return (
    <div className="App">
      <h1>Text Generator using GPT-4</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Enter a prompt:
          <input type="text" value={prompt} onChange={handleChange} />
        </label>
        <button type="submit">Generate Text</button>
      </form>
      {generatedText && <p>Generated text: {generatedText}</p>}
    </div>
  );
}

export default App;
