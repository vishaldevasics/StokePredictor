import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const LoginSignup = () => {
  const [isSignup, setIsSignup] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    fname: '',
    lname: '',
    phonenumber: ''
  });

  const navigate = useNavigate();

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

      const userId = res.data.data.user._id;
      localStorage.setItem('userId', userId);

      alert(`Success: ${res.data.message}`);
      navigate('/predict');
    } catch (err) {
      alert(err.response?.data?.message || 'Something went wrong');
    }
  };

  return (
    <div className="h-screen w-screen flex items-center justify-center bg-gradient-to-br p-4">
      <div className="w-full max-w-md">
        <div className="bg-white bg-opacity-20 backdrop-filter backdrop-blur-lg rounded-3xl shadow-lg overflow-hidden border border-white border-opacity-30">
          <div className="p-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-orange-600 mb-2">
                {isSignup ? '👋 Create Account' : '🔑 Welcome Back'}
              </h2>
              <p className="text-orange-500">
                {isSignup ? 'Join our medical community' : 'Sign in to continue your health journey'}
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5">
              {isSignup && (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <input
                      type="text"
                      name="fname"
                      placeholder="First Name 🧑"
                      value={formData.fname}
                      onChange={handleChange}
                      className="w-full px-4 py-3 rounded-xl bg-white bg-opacity-70 border border-orange-200 focus:outline-none focus:ring-2 focus:ring-orange-400 text-orange-800 placeholder-orange-400"
                      required
                    />
                  </div>
                  <div>
                    <input
                      type="text"
                      name="lname"
                      placeholder="Last Name 👪"
                      value={formData.lname}
                      onChange={handleChange}
                      className="w-full px-4 py-3 rounded-xl bg-white bg-opacity-70 border border-orange-200 focus:outline-none focus:ring-2 focus:ring-orange-400 text-orange-800 placeholder-orange-400"
                      required
                    />
                  </div>
                </div>
              )}

              {isSignup && (
                <div>
                  <input
                    type="text"
                    name="phonenumber"
                    placeholder="Phone Number 📱"
                    value={formData.phonenumber}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-xl bg-white bg-opacity-70 border border-orange-200 focus:outline-none focus:ring-2 focus:ring-orange-400 text-orange-800 placeholder-orange-400"
                  />
                </div>
              )}

              <div>
                <input
                  type="email"
                  name="email"
                  placeholder="Email 📧"
                  value={formData.email}
                  onChange={handleChange}
                  className="w-full px-4 py-3 rounded-xl bg-white bg-opacity-70 border border-orange-200 focus:outline-none focus:ring-2 focus:ring-orange-400 text-orange-800 placeholder-orange-400"
                  required
                />
              </div>

              <div>
                <input
                  type="password"
                  name="password"
                  placeholder="Password 🔒"
                  value={formData.password}
                  onChange={handleChange}
                  className="w-full px-4 py-3 rounded-xl bg-white bg-opacity-70 border border-orange-200 focus:outline-none focus:ring-2 focus:ring-orange-400 text-orange-800 placeholder-orange-400"
                  required
                />
              </div>

              <div>
                <button
                  type="submit"
                  className="w-full py-3 px-4 bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white font-semibold rounded-xl shadow-md transition duration-300 transform hover:scale-105"
                >
                  {isSignup ? 'Sign Up 🚀' : 'Login →'}
                </button>
              </div>
            </form>

            <div className="mt-6 text-center">
              <p className="text-orange-600">
                {isSignup ? 'Already have an account?' : "Don't have an account?"}
                <button
                  onClick={toggleMode}
                  className="ml-2 font-semibold text-orange-700 hover:text-orange-800 underline hover:underline-offset-4 transition-all"
                >
                  {isSignup ? 'Login here' : 'Sign up now'}
                </button>
              </p>
            </div>

            {!isSignup && (
              <div className="mt-6 text-center">
                <button className="text-sm text-orange-600 hover:text-orange-800">
                  Forgot password? 🔑
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="mt-6 text-center text-orange-500 text-sm">
          <p>Your health data is always safe with us 🛡️</p>
        </div>
      </div>
    </div>
  );
};

export default LoginSignup;
