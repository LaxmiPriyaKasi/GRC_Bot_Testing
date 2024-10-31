import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App'; // Assuming your main app component is named App.js

// Render the App component into the root element
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
