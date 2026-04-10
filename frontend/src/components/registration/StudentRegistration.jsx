import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const StudentRegistration = () => {
  const [step, setStep] = useState(1); // 1: Form, 2: Face Capture, 3: Success
  
  // Form data
  const [formData, setFormData] = useState({
    enrollment_id: '',
    full_name: '',
    email: '',
    department: '',
    year: '',
    section: ''
  });
  
  // Webcam and capture
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [capturedImages, setCapturedImages] = useState([]);
  const [currentAngle, setCurrentAngle] = useState(0);
  const [isCapturing, setIsCapturing] = useState(false);
  const [countdown, setCountdown] = useState(null);
  
  // Registration state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  
  const captureAngles = [
    { name: 'Front View', instruction: 'Look straight at the camera', angle: 0 },
    { name: 'Left Turn', instruction: 'Turn your head slightly left', angle: 1 },
    { name: 'Right Turn', instruction: 'Turn your head slightly right', angle: 2 },
    { name: 'Slight Tilt', instruction: 'Tilt your head slightly down', angle: 3 }
  ];
  
  useEffect(() => {
    if (step === 2 && !stream) {
      startWebcam();
    }
    
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [step]);
  
  const startWebcam = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      setError('Failed to access camera. Please grant camera permissions.');
    }
  };
  
  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };
  
  const handleFormSubmit = (e) => {
    e.preventDefault();
    
    // Validate form
    if (!formData.enrollment_id || !formData.full_name) {
      setError('Please fill in required fields');
      return;
    }
    
    setError(null);
    setStep(2);
  };
  
  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    // Start countdown
    setIsCapturing(true);
    let count = 3;
    setCountdown(count);
    
    const countdownInterval = setInterval(() => {
      count--;
      setCountdown(count);
      
      if (count === 0) {
        clearInterval(countdownInterval);
        
        // Capture frame
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0);
        
        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg', 0.9);
        
        // Add to captured images
        setCapturedImages(prev => [...prev, imageData]);
        
        // Move to next angle
        if (currentAngle < captureAngles.length - 1) {
          setCurrentAngle(currentAngle + 1);
        }
        
        setCountdown(null);
        setIsCapturing(false);
      }
    }, 1000);
  };
  
  const retake = (index) => {
    const newImages = [...capturedImages];
    newImages.splice(index, 1);
    setCapturedImages(newImages);
    setCurrentAngle(index);
  };
  
  const handleRegister = async () => {
    if (capturedImages.length < 3) {
      setError('Please capture at least 3 face images');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // Prepare form data
      const formDataToSend = new FormData();
      formDataToSend.append('enrollment_id', formData.enrollment_id);
      formDataToSend.append('full_name', formData.full_name);
      if (formData.email) formDataToSend.append('email', formData.email);
      if (formData.department) formDataToSend.append('department', formData.department);
      if (formData.year) formDataToSend.append('year', formData.year);
      if (formData.section) formDataToSend.append('section', formData.section);
      
      // Add face images
      capturedImages.forEach((image, index) => {
        formDataToSend.append('face_images', image);
      });
      
      // Send to backend
      const response = await axios.post('/api/registration/register', formDataToSend);
      
      if (response.data.success) {
        setSuccess(true);
        setStep(3);
        
        // Stop webcam
        if (stream) {
          stream.getTracks().forEach(track => track.stop());
        }
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  const resetForm = () => {
    setStep(1);
    setFormData({
      enrollment_id: '',
      full_name: '',
      email: '',
      department: '',
      year: '',
      section: ''
    });
    setCapturedImages([]);
    setCurrentAngle(0);
    setSuccess(false);
    setError(null);
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-6 text-white">
            <h1 className="text-3xl font-bold">Student Registration</h1>
            <p className="mt-2 opacity-90">Classroom AI System</p>
          </div>
          
          {/* Progress Bar */}
          <div className="bg-gray-100 px-6 py-4">
            <div className="flex items-center justify-between">
              {['Basic Info', 'Face Capture', 'Complete'].map((label, index) => (
                <div key={index} className="flex items-center">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                    step > index + 1 ? 'bg-green-500 text-white' :
                    step === index + 1 ? 'bg-blue-600 text-white' :
                    'bg-gray-300 text-gray-600'
                  }`}>
                    {step > index + 1 ? '✓' : index + 1}
                  </div>
                  <span className={`ml-2 font-medium ${
                    step >= index + 1 ? 'text-gray-800' : 'text-gray-400'
                  }`}>
                    {label}
                  </span>
                  {index < 2 && <div className="w-16 h-1 mx-4 bg-gray-300"></div>}
                </div>
              ))}
            </div>
          </div>
          
          {/* Content */}
          <div className="p-8">
            {error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                {error}
              </div>
            )}
            
            {/* Step 1: Basic Information */}
            {step === 1 && (
              <form onSubmit={handleFormSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Enrollment ID <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="text"
                      name="enrollment_id"
                      value={formData.enrollment_id}
                      onChange={handleInputChange}
                      required
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="e.g., CS2024001"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Full Name <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="text"
                      name="full_name"
                      value={formData.full_name}
                      onChange={handleInputChange}
                      required
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="John Doe"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Email
                    </label>
                    <input
                      type="email"
                      name="email"
                      value={formData.email}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="john@university.edu"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Department
                    </label>
                    <select
                      name="department"
                      value={formData.department}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="">Select Department</option>
                      <option value="Computer Science">Computer Science</option>
                      <option value="Electronics">Electronics</option>
                      <option value="Mechanical">Mechanical</option>
                      <option value="Civil">Civil</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Year
                    </label>
                    <select
                      name="year"
                      value={formData.year}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="">Select Year</option>
                      <option value="1">1st Year</option>
                      <option value="2">2nd Year</option>
                      <option value="3">3rd Year</option>
                      <option value="4">4th Year</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Section
                    </label>
                    <input
                      type="text"
                      name="section"
                      value={formData.section}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="A"
                    />
                  </div>
                </div>
                
                <button
                  type="submit"
                  className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition"
                >
                  Continue to Face Capture
                </button>
              </form>
            )}
            
            {/* Step 2: Face Capture */}
            {step === 2 && (
              <div className="space-y-6">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h3 className="font-semibold text-blue-900 mb-2">
                    📸 Capture Instructions
                  </h3>
                  <p className="text-blue-800 text-sm">
                    We need to capture your face from different angles for accurate recognition.
                    Follow the on-screen instructions and click "Capture" for each pose.
                  </p>
                </div>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Live Webcam */}
                  <div>
                    <div className="bg-black rounded-lg overflow-hidden relative" style={{ paddingBottom: '75%' }}>
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="absolute top-0 left-0 w-full h-full object-cover"
                      />
                      
                      {/* Countdown Overlay */}
                      {countdown !== null && (
                        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                          <div className="text-white text-8xl font-bold">{countdown}</div>
                        </div>
                      )}
                      
                      {/* Current Instruction */}
                      {currentAngle < captureAngles.length && (
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-4">
                          <p className="text-white text-center font-semibold text-lg">
                            {captureAngles[currentAngle].instruction}
                          </p>
                        </div>
                      )}
                    </div>
                    
                    <canvas ref={canvasRef} style={{ display: 'none' }} />
                    
                    <div className="mt-4 text-center">
                      {capturedImages.length < captureAngles.length ? (
                        <button
                          onClick={captureImage}
                          disabled={isCapturing}
                          className="px-8 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition disabled:opacity-50"
                        >
                          {isCapturing ? 'Capturing...' : `Capture ${captureAngles[currentAngle].name}`}
                        </button>
                      ) : (
                        <p className="text-green-600 font-semibold">✓ All images captured!</p>
                      )}
                    </div>
                  </div>
                  
                  {/* Captured Images */}
                  <div>
                    <h3 className="font-semibold text-gray-800 mb-4">
                      Captured Images ({capturedImages.length}/{captureAngles.length})
                    </h3>
                    
                    <div className="grid grid-cols-2 gap-4">
                      {captureAngles.map((angle, index) => (
                        <div key={index} className="border-2 border-dashed border-gray-300 rounded-lg p-2">
                          {capturedImages[index] ? (
                            <div className="relative">
                              <img
                                src={capturedImages[index]}
                                alt={angle.name}
                                className="w-full rounded"
                              />
                              <button
                                onClick={() => retake(index)}
                                className="absolute top-1 right-1 bg-red-500 text-white px-2 py-1 rounded text-xs hover:bg-red-600"
                              >
                                Retake
                              </button>
                              <p className="text-xs text-green-600 font-semibold mt-1">✓ {angle.name}</p>
                            </div>
                          ) : (
                            <div className="aspect-square flex items-center justify-center text-gray-400">
                              <span className="text-4xl">📷</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                
                <div className="flex gap-4">
                  <button
                    onClick={() => setStep(1)}
                    className="flex-1 border-2 border-gray-300 text-gray-700 py-3 rounded-lg font-semibold hover:bg-gray-50 transition"
                  >
                    Back
                  </button>
                  
                  <button
                    onClick={handleRegister}
                    disabled={capturedImages.length < 3 || loading}
                    className="flex-1 bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition disabled:opacity-50"
                  >
                    {loading ? 'Registering...' : 'Complete Registration'}
                  </button>
                </div>
              </div>
            )}
            
            {/* Step 3: Success */}
            {step === 3 && success && (
              <div className="text-center py-12">
                <div className="text-green-500 text-6xl mb-4">✓</div>
                <h2 className="text-3xl font-bold text-gray-800 mb-2">Registration Successful!</h2>
                <p className="text-gray-600 mb-8">
                  Student <strong>{formData.full_name}</strong> has been registered successfully.
                </p>
                
                <button
                  onClick={resetForm}
                  className="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 transition"
                >
                  Register Another Student
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StudentRegistration;
