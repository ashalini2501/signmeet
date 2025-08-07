-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Feb 12, 2025 at 10:05 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `sign_meet_new`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `ga_gesture`
--

CREATE TABLE `ga_gesture` (
  `id` int(11) NOT NULL,
  `gesture` varchar(50) NOT NULL,
  `fname` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `ga_gesture`
--

INSERT INTO `ga_gesture` (`id`, `gesture`, `fname`) VALUES
(1, 'hi, welcome', 'f1.csv'),
(2, 'Very Super', 'f2.csv'),
(3, 'How are you?', 'f3.csv'),
(4, 'Call you', 'f4.csv'),
(5, 'Peace', 'f5.csv'),
(6, 'feel very bad', 'f6.csv'),
(7, 'Good Morning', 'f7.csv');

-- --------------------------------------------------------

--
-- Table structure for table `sign_image`
--

CREATE TABLE `sign_image` (
  `id` int(11) NOT NULL,
  `message` varchar(200) NOT NULL,
  `image_file` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `sign_image`
--

INSERT INTO `sign_image` (`id`, `message`, `image_file`) VALUES
(1, 'hi welcome', 'F1.gif'),
(2, 'Super', 'F2.gif'),
(3, 'Thank You', 'F3.gif'),
(4, 'hello', 'F4.gif');

-- --------------------------------------------------------

--
-- Table structure for table `sign_user`
--

CREATE TABLE `sign_user` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL,
  `status` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `sign_user`
--

INSERT INTO `sign_user` (`id`, `name`, `mobile`, `email`, `uname`, `pass`, `status`) VALUES
(1, 'Raj', 9894442716, 'raj@gmail.com', 'raj', '1234', 1),
(2, 'Vijay', 9638527415, 'vijay@gmail.com', 'vijay', '1234', 0);

-- --------------------------------------------------------

--
-- Table structure for table `sign_word`
--

CREATE TABLE `sign_word` (
  `id` int(11) NOT NULL,
  `sign` varchar(50) NOT NULL,
  `language` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `sign_word`
--

