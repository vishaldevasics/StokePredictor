import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom'; // for redirecting

const LoginSignup = () => {
  const [isSignup, setIsSignup] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    fname: '',
    lname: '',
    phonenumber: ''
  });

  const navigate = useNavigate(); // ✅ init navigation

  const toggleMode = () => setIsSignup(!isSignup);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const url = isSignup
      ? 'http://localhost:8000/api/v1/users/register'
      : 'http://localhost:8000/api/v1/users/login';
  
    try {
      const res = await axios.post(url, formData, {
        withCredentials: true,
      });
  
      const userId = res.data.data.user._id; // or however the backend sends it
      // console.log();
      
      localStorage.setItem('userId', userId); // ✅ store userId
  
      alert(`Success: ${res.data.message}`);
      navigate('/predict');
  
    } catch (err) {
      alert(err.response?.data?.message || 'Something went wrong');
    }
  };
  

  return (
    <div className="auth-container">
      <h2>{isSignup ? 'Sign Up' : 'Login'}</h2>
      <form onSubmit={handleSubmit}>
        {isSignup && (
          <>
            <input
              type="text"
              name="fname"
              placeholder="First Name"
              value={formData.fname}
              onChange={handleChange}
              required
            />
            <input
              type="text"
              name="lname"
              placeholder="Last Name"
              value={formData.lname}
              onChange={handleChange}
              required
            />
            <input
              type="text"
              name="phonenumber"
              placeholder="Phone Number"
              value={formData.phonenumber}
              onChange={handleChange}
            />
          </>
        )}
        <input
          type="email"
          name="email"
          placeholder="Email"
          value={formData.email}
          onChange={handleChange}
          required
        />
        <input
          type="password"
          name="password"
          placeholder="Password"
          value={formData.password}
          onChange={handleChange}
          required
        />
        <button type="submit">{isSignup ? 'Sign Up' : 'Login'}</button>
      </form>
      <p onClick={toggleMode} style={{ cursor: 'pointer', color: 'blue', marginTop: '10px' }}>
        {isSignup ? 'Already have an account? Login' : "Don't have an account? Sign Up"}
      </p>
    </div>
  );
};

export default LoginSignup;
