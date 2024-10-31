const express = require('express');
const axios = require('axios'); // Axios for HTTP requests
const app = express();
const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || 'http://backend:8000/predict';

app.use(express.json());

app.post('/api/chat', async (req, res) => {
    const userQuery = req.body.query;

    // Validate incoming query
    if (!userQuery) {
        return res.status(400).json({ error: 'Query is required' });
    }

    try {
        // Forward request to FastAPI backend
        const response = await axios.post(BACKEND_URL, {
            dataframe_split: {
                columns: ["query"],
                data: [[userQuery]]
            }
        });
        
        if (response.status === 200) {
            res.json({ response: response.data }); // Send back the response from FastAPI
        } else {
            res.status(response.status).json({ error: response.data });
        }
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Error communicating with the backend API' });
    }
});

app.get('/', (req, res) => {
    res.send('Welcome to the GRC Chatbot API! Use /api/chat for chatbot queries.');
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
