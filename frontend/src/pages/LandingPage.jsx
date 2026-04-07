import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Shield, CheckCircle, Smartphone, Globe, Lock } from 'lucide-react';

const LandingPage = () => {
    return (
        <div className="relative overflow-hidden">
            {/* Hero Section */}
            <div className="relative pt-32 pb-20 lg:pt-48 lg:pb-32">
                <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-blue-50 to-white -z-10" />
                <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-bl from-indigo-50/50 to-transparent -z-10 blur-3xl" />

                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                    >
                        <div className="inline-flex items-center px-4 py-2 rounded-full bg-blue-50 text-blue-700 text-sm font-medium mb-8">
                            <span className="flex h-2 w-2 rounded-full bg-blue-600 mr-2"></span>
                            v2.0 Now Available with Multi-Modal Detection
                        </div>
                        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-gray-900 mb-8 leading-tight">
                            Detect Deepfakes with <br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">
                                Unmatched Precision
                            </span>
                        </h1>
                        <p className="mt-4 max-w-2xl mx-auto text-xl text-gray-600 mb-10">
                            GuardianAI uses advanced machine learning to analyze text, identifying misinformation and AI-generated content in seconds.
                        </p>
                        <div className="flex justify-center gap-4">
                            <Link
                                to="/dashboard"
                                className="px-8 py-4 text-lg font-semibold text-white bg-gray-900 rounded-full hover:bg-gray-800 transition-all hover:shadow-xl hover:-translate-y-1"
                            >
                                Start Analyzing
                            </Link>
                            <Link
                                to="/register"
                                className="px-8 py-4 text-lg font-semibold text-gray-900 bg-white border border-gray-200 rounded-full hover:bg-gray-50 transition-all hover:shadow-lg hover:-translate-y-1"
                            >
                                Create Account
                            </Link>
                        </div>
                    </motion.div>
                </div>
            </div>

            {/* Features Grid */}
            <div className="py-24 bg-white">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
                        {[
                            {
                                icon: <Shield className="w-8 h-8 text-blue-600" />,
                                title: "Text Analysis",
                                desc: "Analyze Text with our specialized models trained on millions of samples."
                            },
                            {
                                icon: <Globe className="w-8 h-8 text-indigo-600" />,
                                title: "Real-time Detection",
                                desc: "Get instant results with detailed confidence scores and breakdown of potential manipulation."
                            },
                            {
                                icon: <Lock className="w-8 h-8 text-teal-600" />,
                                title: "Privacy First",
                                desc: "Your uploads are processed securely and automatically deleted after analysis. We value your privacy."
                            }
                        ].map((feature, idx) => (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: idx * 0.1 }}
                                className="p-8 rounded-2xl bg-gray-50 hover:bg-blue-50/30 transition-colors border border-gray-100"
                            >
                                <div className="w-14 h-14 bg-white rounded-xl shadow-sm flex items-center justify-center mb-6">
                                    {feature.icon}
                                </div>
                                <h3 className="text-xl font-bold text-gray-900 mb-3">{feature.title}</h3>
                                <p className="text-gray-600 leading-relaxed">{feature.desc}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LandingPage;
