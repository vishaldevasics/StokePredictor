// App.js
import React from 'react';
import './App.css';
import HealthForm from './HealthForm';
import LoginSignup from './LoginSignup';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
    
          <Routes>
            <Route path="/" element={<Navigate to="/auth" />} />
            <Route path="/auth" element={<LoginSignup />} />
            <Route path="/predict" element={<HealthForm />} />
          </Routes>
        </header>
      </div>
    </Router>
  );
}

export default App;
