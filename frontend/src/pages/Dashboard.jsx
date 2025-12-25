import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Image as ImageIcon, Video, Mic, FileText, ArrowRight } from 'lucide-react';

const Dashboard = () => {
    const options = [
        {
            id: 'image',
            title: 'Image Analysis',
            desc: 'Detect AI-generated images and deepfakes using visual artifact analysis.',
            icon: <ImageIcon className="w-8 h-8" />,
            color: 'bg-blue-500',
            gradient: 'from-blue-500 to-cyan-500',
            path: '/detect/image'
        },
        {
            id: 'video',
            title: 'Video Analysis',
            desc: 'Analyze video footage for face manipulation and synthetic generation.',
            icon: <Video className="w-8 h-8" />,
            color: 'bg-indigo-500',
            gradient: 'from-indigo-500 to-purple-500',
            path: '/detect/video'
        },
        {
            id: 'audio',
            title: 'Audio Analysis',
            desc: 'Identify synthetic voices and AI-generated audio clips.',
            icon: <Mic className="w-8 h-8" />,
            color: 'bg-rose-500',
            gradient: 'from-rose-500 to-pink-500',
            path: '/detect/audio'
        },
        {
            id: 'text',
            title: 'Text Analysis',
            desc: 'Detect machine-generated text and potential misinformation.',
            icon: <FileText className="w-8 h-8" />,
            color: 'bg-emerald-500',
            gradient: 'from-emerald-500 to-teal-500',
            path: '/detect/text'
        }
    ];

    return (
        <div className="pt-24 pb-12 min-h-screen bg-gray-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="mb-12">
                    <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
                    <p className="mt-2 text-gray-600">Select an analysis tool to get started</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {options.map((option, idx) => (
                        <Link key={option.id} to={option.path}>
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: idx * 0.1 }}
                                whileHover={{ y: -5 }}
                                className="group relative bg-white rounded-2xl p-6 shadow-sm border border-gray-100 h-full overflow-hidden hover:shadow-xl transition-all duration-300"
                            >
                                <div className={`absolute top-0 right-0 w-24 h-24 bg-gradient-to-br ${option.gradient} opacity-10 rounded-bl-full -mr-4 -mt-4 transition-all group-hover:scale-110`} />

                                <div className={`w-14 h-14 rounded-xl mb-6 flex items-center justify-center text-white bg-gradient-to-br ${option.gradient} shadow-lg shadow-${option.color}/20`}>
                                    {option.icon}
                                </div>

                                <h3 className="text-xl font-bold text-gray-900 mb-2">{option.title}</h3>
                                <p className="text-gray-600 text-sm mb-6 leading-relaxed">
                                    {option.desc}
                                </p>

                                <div className="flex items-center text-sm font-medium text-gray-900 group-hover:text-blue-600 transition-colors">
                                    Start Analysis <ArrowRight className="w-4 h-4 ml-2 transition-transform group-hover:translate-x-1" />
                                </div>
                            </motion.div>
                        </Link>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
