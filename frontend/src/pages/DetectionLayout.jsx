import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileText, Image as ImageIcon, Video, Mic, ArrowLeft, TriangleAlert, CircleCheck, Shield } from 'lucide-react';
import api from '../api';
import toast from 'react-hot-toast';
import Chatbot from '../components/Chatbot';

const DetectionLayout = () => {
    const { type } = useParams();
    const navigate = useNavigate();
    const [file, setFile] = useState(null);
    const [text, setText] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [preview, setPreview] = useState(null);

    const config = {
        image: { icon: <ImageIcon className="w-6 h-6" />, title: 'Image Analysis', accept: 'image/*', endpoint: '/detect/image' },
        // video: { icon: <Video className="w-6 h-6" />, title: 'Video Analysis', accept: 'video/*', endpoint: '/detect/video' },
        // audio: { icon: <Mic className="w-6 h-6" />, title: 'Audio Analysis', accept: 'audio/*', endpoint: '/detect/audio' },
        // text: { icon: <FileText className="w-6 h-6" />, title: 'Text Analysis', accept: null, endpoint: '/detect/text' },
    };

    const currentConfig = config[type];

    if (!currentConfig) {
        return <div className="text-center pt-20">Invalid detection type</div>;
    }

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            if (type === 'image') {
                const reader = new FileReader();
                reader.onloadend = () => setPreview(reader.result);
                reader.readAsDataURL(selectedFile);
            } else if (type === 'video' || type === 'audio') {
                setPreview(URL.createObjectURL(selectedFile));
            }
            setResult(null);
        }
    };

    const handleAnalyze = async () => {
        if (type !== 'text' && !file) return toast.error('Please upload a file');
        if (type === 'text' && !text) return toast.error('Please enter text');

        setLoading(true);
        setResult(null);

        const formData = new FormData();
        if (type === 'text') {
            // JSON body for text
        } else {
            formData.append('file', file);
        }

        try {
            let response;
            if (type === 'text') {
                response = await api.post(currentConfig.endpoint, { text });
            } else {
                response = await api.post(currentConfig.endpoint, formData, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                });
            }
            setResult(response.data);
            toast.success('Analysis complete!');
        } catch (error) {
            toast.error('Analysis failed. Please try again.');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="pt-24 pb-12 min-h-screen bg-gray-50">
            <div className="max-w-4xl mx-auto px-4">
                <button
                    onClick={() => navigate('/dashboard')}
                    className="flex items-center text-gray-600 hover:text-gray-900 mb-6 transition-colors"
                >
                    <ArrowLeft className="w-4 h-4 mr-2" /> Back to Dashboard
                </button>

                <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
                    <div className="p-6 border-b border-gray-100 bg-gray-50/50 flex items-center">
                        <div className="p-2 bg-blue-100 text-blue-600 rounded-lg mr-4">
                            {currentConfig.icon}
                        </div>
                        <h1 className="text-2xl font-bold text-gray-900">{currentConfig.title}</h1>
                    </div>

                    <div className="p-8">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            {/* Input Section */}
                            <div className="space-y-6">
                                {type === 'text' ? (
                                    <textarea
                                        className="w-full h-64 p-4 rounded-xl border border-gray-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none resize-none"
                                        placeholder="Paste the text you want to analyze here..."
                                        value={text}
                                        onChange={(e) => setText(e.target.value)}
                                    />
                                ) : (
                                    <div className="border-2 border-dashed border-gray-200 rounded-xl p-8 text-center hover:border-blue-400 transition-colors bg-gray-50/50">
                                        <input
                                            type="file"
                                            id="file-upload"
                                            className="hidden"
                                            accept={currentConfig.accept}
                                            onChange={handleFileChange}
                                        />
                                        <label
                                            htmlFor="file-upload"
                                            className="cursor-pointer flex flex-col items-center justify-center"
                                        >
                                            <Upload className="w-12 h-12 text-gray-400 mb-4" />
                                            <span className="text-sm font-medium text-gray-900">
                                                {file ? file.name : 'Click to upload or drag and drop'}
                                            </span>
                                            <span className="text-xs text-gray-500 mt-2">
                                                {type === 'image' ? 'JPG, PNG, WEBP' : type === 'video' ? 'MP4, MOV' : 'MP3, WAV'}
                                            </span>
                                        </label>
                                    </div>
                                )}

                                {/* Previews */}
                                {preview && type === 'image' && (
                                    <img src={preview} alt="Preview" className="w-full h-48 object-cover rounded-lg" />
                                )}
                                {preview && type === 'video' && (
                                    <video src={preview} controls className="w-full h-48 rounded-lg" />
                                )}
                                {preview && type === 'audio' && (
                                    <audio src={preview} controls className="w-full mt-2" />
                                )}

                                <button
                                    onClick={handleAnalyze}
                                    disabled={loading || (!file && !text)}
                                    className={`w-full py-4 rounded-xl font-bold text-white transition-all transform hover:scale-[1.02] active:scale-[0.98] ${loading || (!file && !text)
                                        ? 'bg-gray-300 cursor-not-allowed'
                                        : 'bg-gradient-to-r from-blue-600 to-indigo-600 shadow-lg hover:shadow-blue-500/30'
                                        }`}
                                >
                                    {loading ? 'Analyzing...' : 'Analyze Content'}
                                </button>
                            </div>

                            {/* Results Section */}
                            <div className="relative min-h-[300px] flex flex-col justify-center">
                                <AnimatePresence>
                                    {result ? (
                                        <motion.div
                                            initial={{ opacity: 0, x: 20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            className={`p-8 rounded-2xl border-2 text-center ${result.fake_prob > 0.5
                                                ? 'bg-red-50 border-red-500 text-red-700'
                                                : 'bg-green-50 border-green-500 text-green-700'
                                                }`}
                                        >
                                            <div className="mb-6 flex justify-center">
                                                {result.fake_prob > 0.5 ? (
                                                    <TriangleAlert className="w-16 h-16" />
                                                ) : (
                                                    <CircleCheck className="w-16 h-16" />
                                                )}
                                            </div>
                                            <h2 className="text-3xl font-extrabold mb-2">
                                                {result.fake_prob > 0.5 ? 'FAKE DETECTED' : 'LIKELY REAL'}
                                            </h2>
                                            <div className="text-5xl font-black mb-4">
                                                {(Math.max(result.fake_prob, result.real_prob) * 100).toFixed(1)}%
                                            </div>
                                            <p className="font-medium opacity-90">
                                                Confidence Score
                                            </p>
                                        </motion.div>
                                    ) : (
                                        <div className="text-center text-gray-400 p-8 border border-gray-100 rounded-2xl bg-gray-50">
                                            <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                                                <Shield className="w-8 h-8 text-gray-300" />
                                            </div>
                                            <p>Results will appear here after analysis</p>
                                        </div>
                                    )}
                                </AnimatePresence>

                                {loading && (
                                    <div className="absolute inset-0 bg-white/80 flex items-center justify-center z-10 rounded-2xl">
                                        <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <Chatbot />
        </div >
    );
};

export default DetectionLayout;
