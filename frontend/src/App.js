import React, { useState } from 'react';
import axios from 'axios';

function App() {
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState('');


    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const res = await axios.post('http://localhost:8000/predict', { 
                dataframe_split: { 
                    columns: ["query"], 
                    data: [[query]] 
                } 
            });
            setResponse(res.data); // Assuming FastAPI returns { response: ... }
        } catch (error) {
            console.error("Error communicating with the backend:", error);
            setResponse('Error communicating with the backend');
        }
    };
    

    return (
        <div>
            <h1>Welcome to My Chatbot!</h1>
            <form id="chat-form" onSubmit={handleSubmit}>
                <input
                    type="text"
                    placeholder="Ask me anything..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                />
                <button type="submit">Send</button>
            </form>
            {response && <div><h2>Response:</h2><p>{response}</p></div>}
        </div>
    );
}

export default App;
