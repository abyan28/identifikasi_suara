-- phpMyAdmin SQL Dump
-- version 5.0.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 28, 2023 at 01:38 PM
-- Server version: 10.4.11-MariaDB
-- PHP Version: 7.4.2

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `voice_classification`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `id_admin` int(11) NOT NULL,
  `nama_admin` varchar(50) NOT NULL,
  `username` varchar(100) NOT NULL,
  `password` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`id_admin`, `nama_admin`, `username`, `password`) VALUES
(1, 'Abyan', 'admin', 'voiceid');

-- --------------------------------------------------------

--
-- Table structure for table `dataset`
--

CREATE TABLE `dataset` (
  `id_dataset` int(11) NOT NULL,
  `label` int(50) NOT NULL,
  `nama_file` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `dataset`
--

INSERT INTO `dataset` (`id_dataset`, `label`, `nama_file`) VALUES
(102, 3, '2022-05-08T14_51_43.224Z.wav'),
(103, 3, '2022-05-08T14_51_53.997Z.wav'),
(104, 3, '2022-05-08T14_52_04.712Z.wav'),
(105, 3, '2022-05-08T14_52_07.921Z.wav'),
(106, 3, '2022-05-08T14_52_11.759Z.wav'),
(107, 3, '2022-05-08T14_52_19.321Z.wav'),
(108, 3, '2022-05-08T14_52_30.711Z.wav'),
(109, 3, '2022-05-08T14_52_38.348Z.wav'),
(110, 3, '2022-05-08T14_52_42.305Z.wav'),
(111, 3, '2022-05-08T14_52_45.366Z.wav'),
(112, 3, '2022-05-08T14_52_49.224Z.wav'),
(113, 3, '2022-05-08T14_52_53.034Z.wav'),
(114, 3, '2022-05-08T14_52_56.562Z.wav'),
(115, 3, '2022-05-08T14_53_00.858Z.wav'),
(116, 3, '2022-05-08T14_53_04.201Z.wav'),
(117, 3, '2022-05-08T14_53_07.759Z.wav'),
(118, 3, '2022-05-08T14_53_18.925Z.wav'),
(119, 3, '2022-05-08T14_53_22.868Z.wav'),
(120, 3, '2022-05-08T14_53_26.678Z.wav'),
(121, 3, '2022-05-08T14_53_29.954Z.wav'),
(122, 3, '2022-05-08T14_53_33.460Z.wav'),
(123, 3, '2022-05-08T14_53_37.059Z.wav'),
(124, 3, '2022-05-08T14_53_44.036Z.wav'),
(125, 3, '2022-05-08T14_53_51.085Z.wav'),
(126, 3, '2022-05-08T14_54_02.811Z.wav'),
(127, 3, '2022-05-08T14_54_06.950Z.wav'),
(128, 3, '2022-05-08T14_54_10.552Z.wav'),
(129, 3, '2022-05-08T14_54_14.762Z.wav'),
(130, 3, '2022-05-08T14_54_19.141Z.wav'),
(131, 3, '2022-05-08T14_54_23.622Z.wav'),
(132, 3, '2022-05-08T14_54_27.236Z.wav'),
(133, 3, '2022-05-08T14_54_40.936Z.wav'),
(134, 3, '2022-05-08T14_54_45.152Z.wav'),
(135, 3, '2022-05-08T14_54_49.421Z.wav'),
(136, 3, '2022-05-08T14_54_53.453Z.wav'),
(137, 3, '2022-05-08T14_55_18.610Z.wav'),
(138, 3, '2022-05-08T14_55_21.788Z.wav'),
(139, 3, '2022-05-08T14_55_25.788Z.wav'),
(140, 3, '2022-05-08T14_55_40.307Z.wav'),
(141, 3, '2022-05-08T15_01_55.386Z.wav'),
(142, 3, '2022-05-08T15_01_59.782Z.wav'),
(143, 3, '2022-05-08T15_02_03.951Z.wav'),
(144, 3, '2022-05-08T15_02_07.604Z.wav'),
(145, 3, '2022-05-08T15_02_11.837Z.wav'),
(146, 3, '2022-05-08T15_02_15.676Z.wav'),
(147, 3, '2022-05-08T15_02_23.163Z.wav'),
(148, 3, '2022-05-08T15_02_27.180Z.wav'),
(149, 3, '2022-05-08T15_02_31.167Z.wav'),
(150, 3, '2022-05-08T15_02_39.463Z.wav'),
(151, 3, '2022-05-08T15_02_47.652Z.wav'),
(152, 4, '2022-05-08T15_45_29.417Z.wav'),
(153, 4, '2022-05-08T15_45_35.387Z.wav'),
(154, 4, '2022-05-08T15_45_41.454Z.wav'),
(155, 4, '2022-05-08T15_45_48.452Z.wav'),
(156, 4, '2022-05-08T15_45_58.203Z.wav'),
(157, 4, '2022-05-08T15_46_02.602Z.wav'),
(158, 4, '2022-05-08T15_46_08.544Z.wav'),
(159, 4, '2022-05-08T15_46_13.018Z.wav'),
(160, 4, '2022-05-08T15_46_18.497Z.wav'),
(161, 4, '2022-05-08T15_46_22.673Z.wav'),
(162, 4, '2022-05-08T15_46_27.020Z.wav'),
(163, 4, '2022-05-08T15_46_31.363Z.wav'),
(164, 4, '2022-05-08T15_46_35.715Z.wav'),
(165, 4, '2022-05-08T15_46_40.671Z.wav'),
(166, 4, '2022-05-08T15_46_45.548Z.wav'),
(167, 4, '2022-05-08T15_46_50.082Z.wav'),
(168, 4, '2022-05-08T15_46_54.082Z.wav'),
(169, 4, '2022-05-08T15_47_07.965Z.wav'),
(170, 4, '2022-05-08T15_47_12.023Z.wav'),
(171, 4, '2022-05-08T15_47_15.468Z.wav'),
(172, 4, '2022-05-08T15_47_18.943Z.wav'),
(173, 4, '2022-05-08T15_47_23.021Z.wav'),
(174, 4, '2022-05-08T15_47_35.199Z.wav'),
(175, 4, '2022-05-08T15_47_39.150Z.wav'),
(176, 4, '2022-05-08T15_47_43.585Z.wav'),
(177, 4, '2022-05-08T15_47_48.036Z.wav'),
(178, 4, '2022-05-08T15_47_52.560Z.wav'),
(179, 4, '2022-05-08T15_48_00.614Z.wav'),
(180, 4, '2022-05-08T15_48_09.715Z.wav'),
(181, 4, '2022-05-08T15_48_23.299Z.wav'),
(182, 4, '2022-05-08T15_48_37.496Z.wav'),
(183, 4, '2022-05-08T15_48_42.393Z.wav'),
(184, 4, '2022-05-08T15_48_47.610Z.wav'),
(185, 4, '2022-05-08T15_48_56.526Z.wav'),
(186, 4, '2022-05-08T15_49_00.813Z.wav'),
(187, 4, '2022-05-08T15_49_14.475Z.wav'),
(188, 4, '2022-05-08T15_49_28.169Z.wav'),
(189, 4, '2022-05-08T15_49_44.925Z.wav'),
(190, 4, '2022-05-08T15_55_15.284Z.wav'),
(191, 4, '2022-05-08T15_55_18.604Z.wav'),
(192, 4, '2022-05-08T15_55_22.243Z.wav'),
(193, 4, '2022-05-08T15_55_25.351Z.wav'),
(194, 4, '2022-05-08T15_55_28.707Z.wav'),
(195, 4, '2022-05-08T15_55_32.215Z.wav'),
(196, 4, '2022-05-08T15_55_35.609Z.wav'),
(197, 4, '2022-05-08T15_55_39.084Z.wav'),
(198, 4, '2022-05-08T15_55_42.343Z.wav'),
(199, 4, '2022-05-08T15_55_45.617Z.wav'),
(200, 4, '2022-05-08T15_55_51.686Z.wav'),
(201, 4, '2022-05-08T15_55_55.080Z.wav'),
(253, 2, '2023-04-30T23_48_00.170Z.wav'),
(254, 2, '2023-04-30T23_48_08.309Z.wav'),
(255, 2, '2023-04-30T23_48_11.454Z.wav'),
(256, 2, '2023-04-30T23_48_46.810Z.wav'),
(257, 2, '2023-04-30T23_48_50.140Z.wav'),
(258, 2, '2023-04-30T23_48_53.804Z.wav'),
(259, 2, '2023-04-30T23_48_57.310Z.wav'),
(260, 2, '2023-04-30T23_49_00.974Z.wav'),
(261, 2, '2023-04-30T23_49_04.596Z.wav'),
(262, 2, '2023-04-30T23_49_08.146Z.wav'),
(263, 2, '2023-04-30T23_49_11.851Z.wav'),
(264, 2, '2023-04-30T23_49_15.453Z.wav'),
(265, 2, '2023-04-30T23_49_19.412Z.wav'),
(266, 2, '2023-04-30T23_49_23.080Z.wav'),
(267, 2, '2023-04-30T23_49_31.080Z.wav'),
(268, 2, '2023-04-30T23_49_34.887Z.wav'),
(269, 2, '2023-04-30T23_49_38.707Z.wav'),
(270, 2, '2023-04-30T23_49_46.901Z.wav'),
(271, 2, '2023-04-30T23_49_50.736Z.wav'),
(272, 2, '2023-04-30T23_49_54.941Z.wav'),
(273, 2, '2023-04-30T23_49_59.159Z.wav'),
(274, 2, '2023-04-30T23_50_03.213Z.wav'),
(275, 2, '2023-04-30T23_50_07.501Z.wav'),
(276, 2, '2023-04-30T23_50_11.963Z.wav'),
(277, 2, '2023-04-30T23_50_25.188Z.wav'),
(278, 2, '2023-04-30T23_50_29.912Z.wav'),
(279, 2, '2023-04-30T23_50_34.334Z.wav'),
(280, 2, '2023-04-30T23_50_38.874Z.wav'),
(281, 2, '2023-04-30T23_51_40.211Z.wav'),
(282, 2, '2023-04-30T23_51_44.184Z.wav'),
(283, 2, '2023-04-30T23_51_48.157Z.wav'),
(284, 2, '2023-04-30T23_51_56.021Z.wav'),
(285, 2, '2023-04-30T23_51_59.841Z.wav'),
(286, 2, '2023-04-30T23_52_07.630Z.wav'),
(287, 2, '2023-04-30T23_52_11.654Z.wav'),
(288, 2, '2023-04-30T23_52_15.753Z.wav'),
(290, 2, '2023-04-30T23_52_28.466Z.wav'),
(291, 2, '2023-04-30T23_52_32.640Z.wav'),
(292, 2, '2023-04-30T23_52_40.712Z.wav'),
(293, 2, '2023-04-30T23_52_44.826Z.wav'),
(294, 2, '2023-04-30T23_52_49.086Z.wav'),
(295, 2, '2023-04-30T23_52_53.612Z.wav'),
(296, 2, '2023-04-30T23_54_00.089Z.wav'),
(297, 2, '2023-05-01T00_00_40.569Z.wav'),
(298, 2, '2023-05-01T00_00_44.093Z.wav'),
(299, 2, '2023-05-01T00_00_47.842Z.wav'),
(300, 2, '2023-05-01T00_00_51.529Z.wav'),
(301, 2, '2023-05-01T00_00_55.301Z.wav'),
(302, 2, '2023-05-01T00_00_58.933Z.wav'),
(304, 1, '2023-05-01T01_34_14.775Z.wav'),
(305, 1, '2023-05-01T01_34_17.793Z.wav'),
(306, 1, '2023-05-01T01_34_21.105Z.wav'),
(307, 1, '2023-05-01T01_34_24.286Z.wav'),
(308, 1, '2023-05-01T01_34_27.552Z.wav'),
(309, 1, '2023-05-01T01_34_34.281Z.wav'),
(310, 1, '2023-05-01T01_34_37.530Z.wav'),
(311, 1, '2023-05-01T01_34_40.858Z.wav'),
(312, 1, '2023-05-01T01_34_44.222Z.wav'),
(313, 1, '2023-05-01T01_34_47.719Z.wav'),
(314, 1, '2023-05-01T01_34_51.056Z.wav'),
(315, 1, '2023-05-01T01_34_54.587Z.wav'),
(316, 1, '2023-05-01T01_34_57.929Z.wav'),
(317, 1, '2023-05-01T01_35_02.368Z.wav'),
(318, 1, '2023-05-01T01_35_06.121Z.wav'),
(319, 1, '2023-05-01T01_35_13.167Z.wav'),
(320, 1, '2023-05-01T01_35_16.617Z.wav'),
(321, 1, '2023-05-01T01_35_20.016Z.wav'),
(322, 1, '2023-05-01T01_35_23.597Z.wav'),
(323, 1, '2023-05-01T01_35_27.183Z.wav'),
(324, 1, '2023-05-01T01_35_30.739Z.wav'),
(325, 1, '2023-05-01T01_35_34.269Z.wav'),
(326, 1, '2023-05-01T01_35_41.306Z.wav'),
(327, 1, '2023-05-01T01_35_44.807Z.wav'),
(328, 1, '2023-05-01T01_35_48.291Z.wav'),
(329, 1, '2023-05-01T01_35_51.557Z.wav'),
(330, 1, '2023-05-01T01_35_54.803Z.wav'),
(331, 1, '2023-05-01T01_36_24.652Z.wav'),
(332, 1, '2023-05-01T01_36_27.731Z.wav'),
(333, 1, '2023-05-01T01_36_30.666Z.wav'),
(334, 1, '2023-05-01T01_36_33.649Z.wav'),
(335, 1, '2023-05-01T01_36_36.646Z.wav'),
(336, 1, '2023-05-01T01_36_39.793Z.wav'),
(337, 1, '2023-05-01T01_36_42.897Z.wav'),
(338, 1, '2023-05-01T01_36_45.911Z.wav'),
(340, 1, '2023-05-01T01_36_52.293Z.wav'),
(341, 1, '2023-05-01T01_36_55.489Z.wav'),
(342, 1, '2023-05-01T01_36_58.807Z.wav'),
(343, 1, '2023-05-01T01_37_04.991Z.wav'),
(344, 1, '2023-05-01T01_37_12.372Z.wav'),
(345, 1, '2023-05-01T01_37_15.722Z.wav'),
(346, 1, '2023-05-01T01_37_19.155Z.wav'),
(347, 1, '2023-05-01T01_37_22.975Z.wav'),
(348, 1, '2023-05-01T01_37_26.614Z.wav'),
(349, 1, '2023-05-01T01_37_30.012Z.wav'),
(350, 1, '2023-05-01T01_37_33.633Z.wav'),
(351, 1, '2023-05-01T01_37_37.015Z.wav'),
(352, 1, '2023-05-01T01_37_40.769Z.wav'),
(353, 1, '2023-05-01T01_37_49.616Z.wav'),
(354, 1, '2023-05-01T01_54_52.531Z.wav'),
(356, 9, '2023-05-02T13_21_23.398Z.wav'),
(357, 9, '2023-05-02T13_21_28.356Z.wav'),
(358, 9, '2023-05-02T13_21_32.499Z.wav'),
(359, 9, '2023-05-02T13_21_36.219Z.wav'),
(360, 9, '2023-05-02T13_21_39.991Z.wav'),
(361, 9, '2023-05-02T13_21_43.731Z.wav'),
(362, 9, '2023-05-02T13_21_47.549Z.wav'),
(363, 9, '2023-05-02T13_21_51.121Z.wav'),
(364, 9, '2023-05-02T13_21_54.776Z.wav'),
(365, 9, '2023-05-02T13_21_59.307Z.wav'),
(366, 9, '2023-05-02T13_23_03.211Z.wav'),
(367, 9, '2023-05-02T13_23_06.768Z.wav'),
(368, 9, '2023-05-02T13_23_10.314Z.wav'),
(369, 9, '2023-05-02T13_23_13.615Z.wav'),
(370, 9, '2023-05-02T13_23_17.158Z.wav'),
(371, 9, '2023-05-02T13_23_20.456Z.wav'),
(372, 9, '2023-05-02T13_23_23.786Z.wav'),
(373, 9, '2023-05-02T13_23_27.276Z.wav'),
(374, 9, '2023-05-02T13_23_30.776Z.wav'),
(375, 9, '2023-05-02T13_23_34.141Z.wav'),
(376, 9, '2023-05-02T13_23_40.859Z.wav'),
(377, 9, '2023-05-02T13_23_44.275Z.wav'),
(378, 9, '2023-05-02T13_23_47.921Z.wav'),
(379, 9, '2023-05-02T13_23_51.211Z.wav'),
(381, 9, '2023-05-02T13_24_08.722Z.wav'),
(382, 9, '2023-05-02T13_24_12.284Z.wav'),
(383, 9, '2023-05-02T13_24_15.901Z.wav'),
(384, 9, '2023-05-02T13_24_19.218Z.wav'),
(385, 9, '2023-05-02T13_24_22.500Z.wav'),
(386, 9, '2023-05-02T13_24_51.745Z.wav'),
(387, 9, '2023-05-02T13_27_01.336Z.wav'),
(388, 9, '2023-05-02T13_27_05.733Z.wav'),
(390, 9, '2023-05-02T13_27_12.892Z.wav'),
(391, 9, '2023-05-02T13_27_16.143Z.wav'),
(392, 9, '2023-05-02T13_27_19.725Z.wav'),
(393, 9, '2023-05-02T13_27_23.400Z.wav'),
(394, 9, '2023-05-02T13_27_26.893Z.wav'),
(395, 9, '2023-05-02T13_27_33.558Z.wav'),
(396, 9, '2023-05-02T13_27_36.779Z.wav'),
(397, 9, '2023-05-02T13_27_43.548Z.wav'),
(398, 9, '2023-05-02T13_27_47.183Z.wav'),
(399, 9, '2023-05-02T13_27_50.763Z.wav'),
(400, 9, '2023-05-02T13_27_57.901Z.wav'),
(401, 9, '2023-05-02T13_28_01.516Z.wav'),
(402, 9, '2023-05-02T13_28_05.101Z.wav'),
(403, 9, '2023-05-02T13_28_08.561Z.wav'),
(404, 9, '2023-05-02T13_28_15.689Z.wav'),
(405, 9, '2023-05-02T13_28_19.161Z.wav'),
(406, 9, '2023-05-02T13_45_55.587Z.wav'),
(407, 9, '2023-05-02T13_46_17.377Z.wav'),
(472, 2, '2023-05-28T09_14_30.370Z.wav');

-- --------------------------------------------------------

--
-- Table structure for table `hyperparam`
--

CREATE TABLE `hyperparam` (
  `id_hyperparam` int(11) NOT NULL,
  `epoch` int(11) NOT NULL,
  `batch_size` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `hyperparam`
--

INSERT INTO `hyperparam` (`id_hyperparam`, `epoch`, `batch_size`) VALUES
(1, 100, 32);

-- --------------------------------------------------------

--
-- Table structure for table `kelas`
--

CREATE TABLE `kelas` (
  `id_kelas` int(11) NOT NULL,
  `nama_kelas` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `kelas`
--

INSERT INTO `kelas` (`id_kelas`, `nama_kelas`) VALUES
(1, 'Abyan'),
(2, 'Ilmi'),
(3, 'Itsbat'),
(4, 'Lana'),
(9, 'Ulya');

-- --------------------------------------------------------

--
-- Table structure for table `log_identifikasi`
--

CREATE TABLE `log_identifikasi` (
  `id_log` int(11) NOT NULL,
  `nama_file_log` varchar(100) NOT NULL,
  `hasil_id` varchar(50) NOT NULL,
  `probabilitas` varchar(11) NOT NULL,
  `tanggal` datetime NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `log_identifikasi`
--

INSERT INTO `log_identifikasi` (`id_log`, `nama_file_log`, `hasil_id`, `probabilitas`, `tanggal`) VALUES
(1, '2022-05-22T155406.676Z.wav', 'Lana', '0', '2022-05-22 22:54:11'),
(2, '2022-05-22T155710.713Z.wav', 'Ulya', '0', '2022-05-22 22:57:16'),
(3, '2022-05-23T005240.129Z.wav', 'Ulya', '0', '2022-05-23 07:52:43'),
(4, '2022-05-23T031955.028Z.wav', 'Ulya', '0', '2022-05-23 10:20:00'),
(6, '2023-04-29T055423.426Z.wav', 'Abyan', '0', '2023-04-29 12:54:26'),
(7, '2023-04-29T055455.645Z.wav', 'Abyan', '0', '2023-04-29 12:54:58'),
(8, '2023-04-29T055656.779Z.wav', 'Lana', '0', '2023-04-29 12:56:59'),
(9, '2023-04-29T055708.226Z.wav', 'Ulya', '0', '2023-04-29 12:57:10'),
(10, '2023-04-29T055720.954Z.wav', 'Ulya', '0', '2023-04-29 12:57:24'),
(11, '2023-04-29T055733.460Z.wav', 'Itsbat', '0', '2023-04-29 12:57:37'),
(12, '2023-04-29T055747.309Z.wav', 'Itsbat', '0', '2023-04-29 12:57:49'),
(13, '2023-04-29T055755.888Z.wav', 'Itsbat', '0', '2023-04-29 12:57:58'),
(14, '2023-04-29T055817.064Z.wav', 'Abyan', '0', '2023-04-29 12:58:19'),
(15, '2023-04-29T055824.333Z.wav', 'Abyan', '0', '2023-04-29 12:58:26'),
(16, '2023-04-29T055831.712Z.wav', 'Abyan', '0', '2023-04-29 12:58:33'),
(17, '2023-04-29T055851.206Z.wav', 'Abyan', '0', '2023-04-29 12:58:53'),
(18, '2023-04-29T055858.995Z.wav', 'Abyan', '0', '2023-04-29 12:59:01'),
(19, '2023-04-29T060627.722Z.wav', 'Ilmi', '0', '2023-04-29 13:06:30'),
(20, '2023-04-29T060635.184Z.wav', 'Ilmi', '0', '2023-04-29 13:06:38'),
(21, '2023-04-29T060723.423Z.wav', 'Ilmi', '0', '2023-04-29 13:07:25'),
(22, '2023-04-29T060742.759Z.wav', 'Itsbat', '0', '2023-04-29 13:07:45'),
(23, '2023-04-29T060751.261Z.wav', 'Itsbat', '0', '2023-04-29 13:07:53'),
(24, '2023-04-29T060908.962Z.wav', 'Itsbat', '0', '2023-04-29 13:09:11'),
(25, '2023-04-29T060927.701Z.wav', 'Itsbat', '0', '2023-04-29 13:09:30'),
(26, '2023-04-29T060935.437Z.wav', 'Itsbat', '0', '2023-04-29 13:09:37'),
(27, '2023-04-29T061312.095Z.wav', 'Itsbat', '0', '2023-04-29 13:13:14'),
(28, '2023-04-29T061320.269Z.wav', 'Ilmi', '0', '2023-04-29 13:13:22'),
(29, '2023-04-29T061328.182Z.wav', 'Abyan', '0', '2023-04-29 13:13:30'),
(30, '2023-04-29T061616.209Z.wav', 'Itsbat', '0', '2023-04-29 13:16:18'),
(31, '2023-04-29T061624.455Z.wav', 'Abyan', '0', '2023-04-29 13:16:26'),
(32, '2023-04-29T061643.765Z.wav', 'Lana', '0', '2023-04-29 13:16:46'),
(33, '2023-04-30T233213.708Z.wav', 'Ilmi', '0', '2023-05-01 06:32:16'),
(34, '2023-04-30T233240.933Z.wav', 'Itsbat', '0', '2023-05-01 06:32:43'),
(35, '2023-04-30T233707.277Z.wav', 'Ilmi', '0', '2023-05-01 06:37:10'),
(36, '2023-04-30T233734.984Z.wav', 'Itsbat', '0', '2023-05-01 06:37:37'),
(37, '2023-04-30T233742.217Z.wav', 'Ilmi', '0', '2023-05-01 06:37:44'),
(38, '2023-04-30T233748.945Z.wav', 'Ilmi', '0', '2023-05-01 06:37:51'),
(39, '2023-04-30T233755.489Z.wav', 'Ilmi', '0', '2023-05-01 06:37:57'),
(40, '2023-04-30T233802.513Z.wav', 'Itsbat', '0', '2023-05-01 06:38:05'),
(41, '2023-04-30T233809.633Z.wav', 'Ilmi', '0', '2023-05-01 06:38:12'),
(42, '2023-04-30T233816.525Z.wav', 'Ilmi', '0', '2023-05-01 06:38:18'),
(43, '2023-04-30T233823.114Z.wav', 'Itsbat', '0', '2023-05-01 06:38:25'),
(44, '2023-04-30T233830.008Z.wav', 'Itsbat', '0', '2023-05-01 06:38:32'),
(45, '2023-04-30T233837.149Z.wav', 'Ilmi', '0', '2023-05-01 06:38:39'),
(46, '2023-04-30T233844.192Z.wav', 'Itsbat', '0', '2023-05-01 06:38:46'),
(47, '2023-04-30T233900.359Z.wav', 'Itsbat', '0', '2023-05-01 06:39:02'),
(48, '2023-04-30T233907.441Z.wav', 'Itsbat', '0', '2023-05-01 06:39:10'),
(49, '2023-04-30T233931.587Z.wav', 'Itsbat', '0', '2023-05-01 06:39:34'),
(50, '2023-04-30T234040.121Z.wav', 'Ilmi', '0', '2023-05-01 06:40:42'),
(51, '2023-04-30T234048.160Z.wav', 'Ilmi', '0', '2023-05-01 06:40:50'),
(52, '2023-04-30T234055.310Z.wav', 'Ilmi', '0', '2023-05-01 06:40:57'),
(53, '2023-04-30T234102.604Z.wav', 'Ilmi', '0', '2023-05-01 06:41:05'),
(54, '2023-04-30T234113.591Z.wav', 'Ilmi', '0', '2023-05-01 06:41:15'),
(55, '2023-04-30T234120.034Z.wav', 'Ilmi', '0', '2023-05-01 06:41:22'),
(56, '2023-04-30T234127.885Z.wav', 'Ilmi', '0', '2023-05-01 06:41:30'),
(57, '2023-04-30T234134.395Z.wav', 'Ilmi', '0', '2023-05-01 06:41:37'),
(58, '2023-04-30T234142.319Z.wav', 'Ilmi', '0', '2023-05-01 06:41:44'),
(59, '2023-04-30T234149.137Z.wav', 'Ilmi', '0', '2023-05-01 06:41:51'),
(60, '2023-04-30T234155.477Z.wav', 'Ilmi', '0', '2023-05-01 06:41:57'),
(61, '2023-04-30T234204.592Z.wav', 'Ilmi', '0', '2023-05-01 06:42:06'),
(62, '2023-04-30T234211.046Z.wav', 'Ilmi', '0', '2023-05-01 06:42:13'),
(63, '2023-04-30T234227.329Z.wav', 'Ilmi', '0', '2023-05-01 06:42:29'),
(64, '2023-04-30T234244.378Z.wav', 'Ilmi', '0', '2023-05-01 06:42:47'),
(65, '2023-04-30T234252.644Z.wav', 'Ilmi', '0', '2023-05-01 06:42:54'),
(66, '2023-04-30T234259.682Z.wav', 'Ilmi', '0', '2023-05-01 06:43:01'),
(67, '2023-04-30T234306.584Z.wav', 'Ilmi', '0', '2023-05-01 06:43:08'),
(68, '2023-04-30T234313.794Z.wav', 'Ilmi', '0', '2023-05-01 06:43:16'),
(69, '2023-04-30T234320.866Z.wav', 'Ilmi', '0', '2023-05-01 06:43:23'),
(70, '2023-04-30T234341.808Z.wav', 'Ilmi', '0', '2023-05-01 06:43:44'),
(71, '2023-04-30T234740.947Z.wav', 'Itsbat', '0', '2023-05-01 06:47:43'),
(72, '2023-05-01T012440.702Z.wav', 'Ilmi', '0', '2023-05-01 08:24:43'),
(73, '2023-05-01T012453.578Z.wav', 'Ilmi', '0', '2023-05-01 08:24:55'),
(74, '2023-05-01T012500.920Z.wav', 'Ilmi', '0', '2023-05-01 08:25:03'),
(75, '2023-05-01T012510.038Z.wav', 'Ilmi', '0', '2023-05-01 08:25:12'),
(76, '2023-05-01T012519.598Z.wav', 'Ilmi', '0', '2023-05-01 08:25:22'),
(77, '2023-05-01T012556.488Z.wav', 'Itsbat', '0', '2023-05-01 08:25:59'),
(78, '2023-05-01T012604.092Z.wav', 'Abyan', '0', '2023-05-01 08:26:07'),
(79, '2023-05-01T012612.246Z.wav', 'Ilmi', '0', '2023-05-01 08:26:15'),
(80, '2023-05-01T012621.356Z.wav', 'Ilmi', '0', '2023-05-01 08:26:23'),
(81, '2023-05-01T012628.917Z.wav', 'Ilmi', '0', '2023-05-01 08:26:31'),
(82, '2023-05-01T012636.673Z.wav', 'Ilmi', '0', '2023-05-01 08:26:39'),
(83, '2023-05-01T012644.330Z.wav', 'Ilmi', '0', '2023-05-01 08:26:46'),
(84, '2023-05-01T012653.783Z.wav', 'Itsbat', '0', '2023-05-01 08:26:56'),
(85, '2023-05-01T012701.836Z.wav', 'Itsbat', '0', '2023-05-01 08:27:04'),
(86, '2023-05-01T012709.622Z.wav', 'Ilmi', '0', '2023-05-01 08:27:11'),
(87, '2023-05-01T012718.846Z.wav', 'Ilmi', '0', '2023-05-01 08:27:21'),
(88, '2023-05-01T012729.811Z.wav', 'Ilmi', '0', '2023-05-01 08:27:31'),
(89, '2023-05-01T012745.244Z.wav', 'Ilmi', '0', '2023-05-01 08:27:48'),
(90, '2023-05-01T012754.420Z.wav', 'Ilmi', '0', '2023-05-01 08:27:56'),
(91, '2023-05-01T012810.988Z.wav', 'Ilmi', '0', '2023-05-01 08:28:13'),
(92, '2023-05-01T012818.554Z.wav', 'Ilmi', '0', '2023-05-01 08:28:21'),
(93, '2023-05-01T013025.281Z.wav', 'Ilmi', '0', '2023-05-01 08:30:27'),
(94, '2023-05-01T013032.394Z.wav', 'Ilmi', '0', '2023-05-01 08:30:34'),
(95, '2023-05-01T013039.452Z.wav', 'Ilmi', '0', '2023-05-01 08:30:41'),
(96, '2023-05-01T013046.468Z.wav', 'Ilmi', '0', '2023-05-01 08:30:48'),
(97, '2023-05-01T013054.024Z.wav', 'Ilmi', '0', '2023-05-01 08:30:56'),
(98, '2023-05-01T013102.583Z.wav', 'Itsbat', '0', '2023-05-01 08:31:04'),
(99, '2023-05-01T013110.999Z.wav', 'Ilmi', '0', '2023-05-01 08:31:13'),
(100, '2023-05-01T013123.383Z.wav', 'Ilmi', '0', '2023-05-01 08:31:25'),
(101, '2023-05-01T013242.877Z.wav', 'Ilmi', '0', '2023-05-01 08:32:45'),
(102, '2023-05-01T013251.179Z.wav', 'Ilmi', '0', '2023-05-01 08:32:53'),
(103, '2023-05-01T032253.764Z.wav', 'Abyan', '0', '2023-05-01 10:22:56'),
(104, '2023-05-01T032300.961Z.wav', 'Abyan', '0', '2023-05-01 10:23:03'),
(105, '2023-05-01T032308.094Z.wav', 'Abyan', '0', '2023-05-01 10:23:11'),
(106, '2023-05-01T032316.388Z.wav', 'Abyan', '0', '2023-05-01 10:23:18'),
(107, '2023-05-01T032354.130Z.wav', 'Ilmi', '0', '2023-05-01 10:23:56'),
(108, '2023-05-01T032404.074Z.wav', 'Ilmi', '0', '2023-05-01 10:24:06'),
(109, '2023-05-01T032411.866Z.wav', 'Ilmi', '0', '2023-05-01 10:24:14'),
(110, '2023-05-01T032419.152Z.wav', 'Ilmi', '0', '2023-05-01 10:24:21'),
(111, '2023-05-01T032425.789Z.wav', 'Ilmi', '0', '2023-05-01 10:24:28'),
(112, '2023-05-01T032433.756Z.wav', 'Ilmi', '0', '2023-05-01 10:24:36'),
(113, '2023-05-01T032708.127Z.wav', 'Abyan', '0', '2023-05-01 10:27:10'),
(114, '2023-05-01T032714.886Z.wav', 'Abyan', '0', '2023-05-01 10:27:17'),
(115, '2023-05-01T032723.014Z.wav', 'Abyan', '0', '2023-05-01 10:27:25'),
(116, '2023-05-01T032732.026Z.wav', 'Abyan', '0', '2023-05-01 10:27:34'),
(117, '2023-05-01T032738.654Z.wav', 'Ilmi', '0', '2023-05-01 10:27:40'),
(118, '2023-05-01T032746.142Z.wav', 'Abyan', '0', '2023-05-01 10:27:48'),
(119, '2023-05-01T032752.383Z.wav', 'Abyan', '0', '2023-05-01 10:27:54'),
(120, '2023-05-01T032759.693Z.wav', 'Abyan', '0', '2023-05-01 10:28:01'),
(121, '2023-05-01T032807.114Z.wav', 'Abyan', '0', '2023-05-01 10:28:09'),
(122, '2023-05-01T032815.995Z.wav', 'Abyan', '0', '2023-05-01 10:28:18'),
(123, '2023-05-01T032823.331Z.wav', 'Abyan', '0', '2023-05-01 10:28:25'),
(124, '2023-05-01T032830.807Z.wav', 'Abyan', '0', '2023-05-01 10:28:33'),
(125, '2023-05-01T032838.334Z.wav', 'Ilmi', '0', '2023-05-01 10:28:40'),
(126, '2023-05-01T032845.485Z.wav', 'Ilmi', '0', '2023-05-01 10:28:47'),
(127, '2023-05-01T032852.357Z.wav', 'Abyan', '0', '2023-05-01 10:28:54'),
(128, '2023-05-01T032859.951Z.wav', 'Abyan', '0', '2023-05-01 10:29:02'),
(129, '2023-05-01T032907.587Z.wav', 'Abyan', '0', '2023-05-01 10:29:09'),
(130, '2023-05-01T032915.016Z.wav', 'Abyan', '0', '2023-05-01 10:29:17'),
(131, '2023-05-01T032924.757Z.wav', 'Abyan', '0', '2023-05-01 10:29:27'),
(132, '2023-05-01T032932.386Z.wav', 'Abyan', '0', '2023-05-01 10:29:34'),
(133, '2023-05-01T033728.429Z.wav', 'Ilmi', '0', '2023-05-01 10:37:51'),
(134, '2023-05-01T033757.608Z.wav', 'Ilmi', '0', '2023-05-01 10:38:00'),
(135, '2023-05-01T033806.014Z.wav', 'Ilmi', '0', '2023-05-01 10:38:08'),
(136, '2023-05-01T033813.258Z.wav', 'Ilmi', '0', '2023-05-01 10:38:15'),
(137, '2023-05-01T033821.805Z.wav', 'Itsbat', '0', '2023-05-01 10:38:24'),
(138, '2023-05-01T033830.144Z.wav', 'Ilmi', '0', '2023-05-01 10:38:32'),
(139, '2023-05-01T033838.097Z.wav', 'Ilmi', '0', '2023-05-01 10:38:40'),
(140, '2023-05-01T033935.963Z.wav', 'Ilmi', '0', '2023-05-01 10:39:40'),
(141, '2023-05-01T033946.779Z.wav', 'Ilmi', '0', '2023-05-01 10:39:49'),
(142, '2023-05-01T033954.724Z.wav', 'Ilmi', '0', '2023-05-01 10:39:57'),
(143, '2023-05-01T034002.210Z.wav', 'Ilmi', '0', '2023-05-01 10:40:04'),
(144, '2023-05-01T034010.702Z.wav', 'Ilmi', '0', '2023-05-01 10:40:13'),
(145, '2023-05-01T034018.809Z.wav', 'Ilmi', '0', '2023-05-01 10:40:21'),
(146, '2023-05-01T034026.988Z.wav', 'Ilmi', '0', '2023-05-01 10:40:29'),
(147, '2023-05-01T034035.957Z.wav', 'Ilmi', '0', '2023-05-01 10:40:38'),
(148, '2023-05-01T034043.757Z.wav', 'Ilmi', '0', '2023-05-01 10:40:46'),
(149, '2023-05-01T034051.697Z.wav', 'Ilmi', '0', '2023-05-01 10:40:54'),
(150, '2023-05-01T034151.523Z.wav', 'Ilmi', '0', '2023-05-01 10:41:53'),
(151, '2023-05-01T034159.597Z.wav', 'Ilmi', '0', '2023-05-01 10:42:02'),
(152, '2023-05-01T034209.742Z.wav', 'Ilmi', '0', '2023-05-01 10:42:11'),
(153, '2023-05-01T113016.743Z.wav', 'Lana', '0', '2023-05-01 18:30:19'),
(154, '2023-05-01T113033.447Z.wav', 'Lana', '0', '2023-05-01 18:30:35'),
(155, '2023-05-01T113041.429Z.wav', 'Lana', '0', '2023-05-01 18:30:43'),
(156, '2023-05-01T113050.057Z.wav', 'Lana', '0', '2023-05-01 18:30:52'),
(157, '2023-05-01T113056.965Z.wav', 'Lana', '0', '2023-05-01 18:31:00'),
(158, '2023-05-01T113105.272Z.wav', 'Lana', '0', '2023-05-01 18:31:07'),
(159, '2023-05-01T113117.775Z.wav', 'Lana', '0', '2023-05-01 18:31:19'),
(160, '2023-05-01T113125.639Z.wav', 'Lana', '0', '2023-05-01 18:31:27'),
(161, '2023-05-01T113133.343Z.wav', 'Lana', '0', '2023-05-01 18:31:35'),
(162, '2023-05-01T113141.169Z.wav', 'Lana', '0', '2023-05-01 18:31:43'),
(163, '2023-05-01T113148.033Z.wav', 'Lana', '0', '2023-05-01 18:31:50'),
(164, '2023-05-01T113154.836Z.wav', 'Itsbat', '0', '2023-05-01 18:31:57'),
(165, '2023-05-01T113203.688Z.wav', 'Lana', '0', '2023-05-01 18:32:05'),
(166, '2023-05-01T113211.278Z.wav', 'Lana', '0', '2023-05-01 18:32:13'),
(167, '2023-05-01T113300.109Z.wav', 'Itsbat', '0', '2023-05-01 18:33:02'),
(168, '2023-05-01T113312.700Z.wav', 'Lana', '0', '2023-05-01 18:33:15'),
(169, '2023-05-01T113320.307Z.wav', 'Lana', '0', '2023-05-01 18:33:22'),
(170, '2023-05-01T113330.151Z.wav', 'Lana', '0', '2023-05-01 18:33:32'),
(171, '2023-05-01T113338.047Z.wav', 'Itsbat', '0', '2023-05-01 18:33:40'),
(172, '2023-05-01T113411.090Z.wav', 'Lana', '0', '2023-05-01 18:34:13'),
(173, '2023-05-01T134558.361Z.wav', 'Abyan', '0', '2023-05-01 20:46:00'),
(174, '2023-05-01T134845.814Z.wav', 'Ilmi', '0', '2023-05-01 20:48:48'),
(175, '2023-05-01T135637.158Z.wav', 'Ilmi', '0', '2023-05-01 20:56:46'),
(176, '2023-05-01T141436.278Z.wav', 'Ilmi', '0', '2023-05-01 21:14:43'),
(177, '2023-05-02T133705.523Z.wav', 'Ilmi', '0', '2023-05-02 20:37:15'),
(178, '2023-05-02T145602.178Z.wav', 'Abyan', '0', '2023-05-02 21:56:05'),
(179, '2023-05-02T145611.743Z.wav', 'Abyan', '0', '2023-05-02 21:56:14'),
(180, '2023-05-02T145716.628Z.wav', 'Ulya', '0', '2023-05-02 21:57:19'),
(181, '2023-05-02T145725.806Z.wav', 'Ulya', '0', '2023-05-02 21:57:28'),
(182, '2023-05-02T145734.410Z.wav', 'Ulya', '0', '2023-05-02 21:57:38'),
(183, '2023-05-02T145743.824Z.wav', 'Ulya', '0', '2023-05-02 21:57:46'),
(184, '2023-05-02T145752.743Z.wav', 'Ulya', '0', '2023-05-02 21:57:55'),
(185, '2023-05-02T145800.816Z.wav', 'Ulya', '0', '2023-05-02 21:58:03'),
(186, '2023-05-02T145808.878Z.wav', 'Ulya', '0', '2023-05-02 21:58:11'),
(187, '2023-05-02T145816.637Z.wav', 'Ulya', '0', '2023-05-02 21:58:18'),
(188, '2023-05-02T145824.769Z.wav', 'Ulya', '0', '2023-05-02 21:58:26'),
(189, '2023-05-02T145832.666Z.wav', 'Ulya', '0', '2023-05-02 21:58:35'),
(190, '2023-05-02T145840.507Z.wav', 'Ulya', '0', '2023-05-02 21:58:42'),
(191, '2023-05-02T145908.513Z.wav', 'Ulya', '0', '2023-05-02 21:59:10'),
(192, '2023-05-02T145916.602Z.wav', 'Ulya', '0', '2023-05-02 21:59:18'),
(193, '2023-05-02T145923.931Z.wav', 'Ulya', '0', '2023-05-02 21:59:26'),
(194, '2023-05-02T145932.565Z.wav', 'Ulya', '0', '2023-05-02 21:59:34'),
(195, '2023-05-02T145939.553Z.wav', 'Ulya', '0', '2023-05-02 21:59:41'),
(196, '2023-05-02T145946.720Z.wav', 'Ulya', '0', '2023-05-02 21:59:48'),
(197, '2023-05-02T145954.496Z.wav', 'Ulya', '0', '2023-05-02 21:59:56'),
(198, '2023-05-02T150003.279Z.wav', 'Ulya', '0', '2023-05-02 22:00:06'),
(199, '2023-05-02T150030.368Z.wav', 'Ulya', '0', '2023-05-02 22:00:32'),
(200, '2023-05-03T043722.804Z.wav', 'Abyan', '0', '2023-05-03 11:37:31'),
(201, '2023-05-03T044108.048Z.wav', 'Ilmi', '0', '2023-05-03 11:41:15'),
(202, '2023-05-03T044125.978Z.wav', 'Abyan', '0', '2023-05-03 11:41:37'),
(203, '2023-05-03T044258.675Z.wav', 'Abyan', '0', '2023-05-03 11:43:03'),
(204, '2023-05-03T044313.750Z.wav', 'Abyan', '0', '2023-05-03 11:43:16'),
(205, '2023-05-03T044324.408Z.wav', 'Itsbat', '0', '2023-05-03 11:43:27'),
(206, '2023-05-03T044333.917Z.wav', 'Itsbat', '0', '2023-05-03 11:43:37'),
(207, '2023-05-03T044346.457Z.wav', 'Itsbat', '0', '2023-05-03 11:43:49'),
(208, '2023-05-03T044355.235Z.wav', 'Abyan', '0', '2023-05-03 11:43:57'),
(209, '2023-05-03T044402.896Z.wav', 'Itsbat', '0', '2023-05-03 11:44:05'),
(210, '2023-05-03T044410.084Z.wav', 'Itsbat', '0', '2023-05-03 11:44:12'),
(211, '2023-05-03T044416.980Z.wav', 'Itsbat', '0', '2023-05-03 11:44:19'),
(212, '2023-05-03T044423.977Z.wav', 'Itsbat', '0', '2023-05-03 11:44:26'),
(213, '2023-05-03T044430.913Z.wav', 'Ilmi', '0', '2023-05-03 11:44:33'),
(214, '2023-05-03T044438.301Z.wav', 'Itsbat', '0', '2023-05-03 11:44:40'),
(215, '2023-05-03T044445.777Z.wav', 'Itsbat', '0', '2023-05-03 11:44:48'),
(216, '2023-05-03T044632.410Z.wav', 'Ilmi', '0', '2023-05-03 11:46:34'),
(217, '2023-05-03T044639.943Z.wav', 'Itsbat', '0', '2023-05-03 11:46:42'),
(218, '2023-05-03T044647.377Z.wav', 'Itsbat', '0', '2023-05-03 11:46:49'),
(219, '2023-05-03T044654.520Z.wav', 'Itsbat', '0', '2023-05-03 11:46:56'),
(220, '2023-05-03T044701.663Z.wav', 'Ilmi', '0', '2023-05-03 11:47:03'),
(221, '2023-05-03T044709.442Z.wav', 'Itsbat', '0', '2023-05-03 11:47:11'),
(222, '2023-05-03T044729.980Z.wav', 'Itsbat', '0', '2023-05-03 11:47:33'),
(223, '2023-05-03T134430.187Z.wav', 'Ilmi', '0', '2023-05-03 20:44:48'),
(224, '2023-05-03T145028.820Z.wav', 'Ilmi', '0', '2023-05-03 21:50:31'),
(225, '2023-05-03T145040.701Z.wav', 'Abyan', '0', '2023-05-03 21:50:43'),
(226, '2023-05-03T150104.455Z.wav', 'Ulya', '0', '2023-05-03 22:01:06'),
(227, '2023-05-03T150118.933Z.wav', 'Abyan', '0', '2023-05-03 22:01:20'),
(228, '2023-05-03T150128.973Z.wav', 'Abyan', '0', '2023-05-03 22:01:35'),
(229, '2023-05-04T014916.906Z.wav', 'Abyan', '0', '2023-05-04 08:49:26'),
(230, '2023-05-05T083843.204Z.wav', 'Ilmi', '0', '2023-05-05 15:38:56'),
(231, '2023-05-05T103103.254Z.wav', 'Ilmi', '0', '2023-05-05 17:31:11'),
(232, '2023-05-05T115535.782Z.wav', 'Ilmi', '0', '2023-05-05 18:56:17'),
(233, '2023-05-06T162032.646Z.wav', 'Abyan', '0', '2023-05-06 23:20:42'),
(234, '2023-05-08T082335.478Z.wav', 'Abyan', '0', '2023-05-08 15:23:46'),
(235, '2023-05-11T052238.497Z.wav', 'Abyan', '0', '2023-05-11 12:22:47'),
(236, '2023-05-11T143247.064Z.wav', 'Itsbat', '0', '2023-05-11 21:32:54'),
(237, '2023-05-11T143302.675Z.wav', 'Ilmi', '0', '2023-05-11 21:33:08'),
(238, '2023-05-11T143315.826Z.wav', 'Ulya', '0', '2023-05-11 21:33:18'),
(239, '2023-05-11T154118.583Z.wav', 'Ulya', '0', '2023-05-11 22:41:20'),
(240, '2023-05-11T154427.078Z.wav', 'Abyan', '0', '2023-05-11 22:44:54'),
(241, '2023-05-11T154506.324Z.wav', 'Abyan', '0', '2023-05-11 22:45:08'),
(242, '2023-05-11T154516.367Z.wav', 'Abyan', '0', '2023-05-11 22:45:17'),
(243, '2023-05-12T025718.852Z.wav', 'Ilmi', '0', '2023-05-12 10:55:19'),
(244, '2023-05-12T054310.010Z.wav', 'Ilmi', '0', '2023-05-12 12:43:17'),
(245, '2023-05-12T063901.929Z.wav', 'Lana', '0', '2023-05-12 13:41:24'),
(246, '2023-05-12T080533.084Z.wav', 'Lana', '0', '2023-05-12 15:05:45'),
(247, '2023-05-12T095005.797Z.wav', 'Lana', '0', '2023-05-12 16:50:13'),
(248, '2023-05-12T100356.612Z.wav', 'Lana', '0', '2023-05-12 17:03:58'),
(249, '2023-05-12T100414.477Z.wav', 'Lana', '0', '2023-05-12 17:04:16'),
(250, '2023-05-12T100513.794Z.wav', 'Lana', '0', '2023-05-12 17:05:14'),
(251, '2023-05-12T100525.662Z.wav', 'Lana', '0', '2023-05-12 17:05:25'),
(252, '2023-05-12T125529.039Z.wav', 'Lana', '0', '2023-05-12 19:55:46'),
(253, '2023-05-12T143244.308Z.wav', 'Itsbat', '0', '2023-05-12 21:32:50'),
(254, '2023-05-12T142243.963Z.wav', 'Itsbat', '0', '2023-05-12 21:37:14'),
(255, '2023-05-12T145544.183Z.wav', 'Lana', '0', '2023-05-12 21:55:48'),
(256, '2023-05-12T145559.287Z.wav', 'Itsbat', '0', '2023-05-12 21:56:10'),
(257, '2023-05-12T152512.721Z.wav', 'Abyan', '0', '2023-05-12 22:25:27'),
(258, '2023-05-12T152618.830Z.wav', 'Abyan', '0', '2023-05-12 22:26:21'),
(259, '2023-05-12T154342.605Z.wav', 'Abyan', '0', '2023-05-12 22:43:46'),
(260, '2023-05-12T155838.754Z.wav', 'Lana', '0', '2023-05-12 22:58:43'),
(261, '2023-05-12T155855.914Z.wav', 'Lana', '0', '2023-05-12 22:59:00'),
(262, '2023-05-13T093728.461Z.wav', 'Lana', '0', '2023-05-13 16:38:06'),
(263, '2023-05-13T132209.704Z.wav', 'Lana', '0', '2023-05-13 20:23:04'),
(264, '2023-05-13T134742.485Z.wav', 'Lana', '0', '2023-05-13 20:48:53'),
(265, '2023-05-20T095239.259Z.wav', 'Abyan', '0.936497', '2023-05-20 16:52:48'),
(266, '2023-05-20T095700.151Z.wav', 'Abyan', '0.919649', '2023-05-20 16:57:06'),
(267, '2023-05-20T100716.144Z.wav', 'Abyan', '99.07', '2023-05-20 17:07:22'),
(268, '2023-05-20T100742.616Z.wav', 'Ilmi', '98.3', '2023-05-20 17:07:44'),
(269, '2023-05-20T100851.082Z.wav', 'Abyan', '97.43', '2023-05-20 17:08:53'),
(270, '2023-05-20T100859.145Z.wav', 'Abyan', '90.56', '2023-05-20 17:09:01'),
(271, '2023-05-20T100923.922Z.wav', 'Lana', '49.37', '2023-05-20 17:09:26'),
(272, '2023-05-20T101632.322Z.wav', 'Ilmi', '52.23', '2023-05-20 17:16:35'),
(273, '2023-05-20T105107.065Z.wav', 'Ilmi', '62.67%', '2023-05-20 17:51:11'),
(274, '2023-05-20T105131.462Z.wav', 'Abyan', '92.16%', '2023-05-20 17:51:35'),
(275, '2023-05-20T105139.425Z.wav', 'Ilmi', '75.10%', '2023-05-20 17:51:41'),
(276, '2023-05-20T105147.861Z.wav', 'Ilmi', '53.39%', '2023-05-20 17:51:49'),
(277, '2023-05-20T122324.208Z.wav', 'Abyan', '99.01%', '2023-05-20 19:23:30'),
(278, '2023-05-20T122436.593Z.wav', 'Dzaka', '83.41%', '2023-05-20 19:24:39'),
(279, '2023-05-20T123752.191Z.wav', 'Ilmi', '96.04%', '2023-05-20 19:37:59'),
(280, '2023-05-20T123844.348Z.wav', 'Not Recognized', '49.84%', '2023-05-20 19:38:46'),
(281, '2023-05-20T123918.111Z.wav', 'Not Recognized', '46.75%', '2023-05-20 19:39:21'),
(282, '2023-05-20T124038.537Z.wav', 'Ilmi', '80.22%', '2023-05-20 19:40:41'),
(283, '2023-05-20T124051.082Z.wav', 'Abyan', '97.49%', '2023-05-20 19:40:53'),
(284, '2023-05-20T124056.951Z.wav', 'Abyan', '99.07%', '2023-05-20 19:40:59'),
(285, '2023-05-20T124102.934Z.wav', 'Abyan', '97.81%', '2023-05-20 19:41:04'),
(286, '2023-05-20T124108.660Z.wav', 'Abyan', '98.48%', '2023-05-20 19:41:10'),
(287, '2023-05-20T134937.182Z.wav', 'Abyan', '93.95%', '2023-05-20 20:49:39'),
(288, '2023-05-20T134948.246Z.wav', 'Ilmi', '65.15%', '2023-05-20 20:49:51'),
(289, '2023-05-20T135000.360Z.wav', 'Ilmi', '85.50%', '2023-05-20 20:50:03'),
(290, '2023-05-20T135020.054Z.wav', 'Lana', '97.80%', '2023-05-20 20:50:23'),
(291, '2023-05-20T135037.693Z.wav', 'Lana', '79.39%', '2023-05-20 20:50:40'),
(292, '2023-05-20T135116.407Z.wav', 'Ilmi', '71.33%', '2023-05-20 20:51:20'),
(293, '2023-05-20T135133.343Z.wav', 'Abyan', '92.14%', '2023-05-20 20:51:35'),
(294, '2023-05-20T135145.111Z.wav', 'Ilmi', '70.87%', '2023-05-20 20:51:47'),
(295, '2023-05-20T135159.979Z.wav', 'Abyan', '72.33%', '2023-05-20 20:52:02'),
(296, '2023-05-26T094537.492Z.wav', 'Abyan', '88.98%', '2023-05-26 16:45:47'),
(297, '2023-05-26T094605.135Z.wav', 'Ilmi', '98.06%', '2023-05-26 16:46:07'),
(298, '2023-05-26T094705.964Z.wav', 'Ilmi', '69.71%', '2023-05-26 16:47:08'),
(299, '2023-05-26T094721.714Z.wav', 'Ilmi', '91.38%', '2023-05-26 16:47:23'),
(300, '2023-05-26T094804.472Z.wav', 'Ilmi', '95.75%', '2023-05-26 16:48:07'),
(301, '2023-05-26T094906.866Z.wav', 'Ilmi', '95.95%', '2023-05-26 16:49:09'),
(302, '2023-05-28T090523.057Z.wav', 'Abyan', '98.49%', '2023-05-28 16:05:33'),
(303, '2023-05-28T090540.211Z.wav', 'Abyan', '98.94%', '2023-05-28 16:05:42'),
(304, '2023-05-28T090708.140Z.wav', 'Ilmi', '54.30%', '2023-05-28 16:07:11'),
(305, '2023-05-28T090725.523Z.wav', 'Abyan', '61.07%', '2023-05-28 16:07:29'),
(306, '2023-05-28T090743.622Z.wav', 'Ilmi', '80.72%', '2023-05-28 16:07:46'),
(307, '2023-05-28T090753.936Z.wav', 'Abyan', '67.61%', '2023-05-28 16:07:56'),
(308, '2023-05-28T090803.082Z.wav', 'Ilmi', '72.18%', '2023-05-28 16:08:05'),
(309, '2023-05-28T090918.458Z.wav', 'Ilmi', '72.58%', '2023-05-28 16:09:22'),
(310, '2023-05-28T090934.506Z.wav', 'Ilmi', '91.05%', '2023-05-28 16:09:36'),
(311, '2023-05-28T090945.946Z.wav', 'Ilmi', '65.50%', '2023-05-28 16:09:48'),
(312, '2023-05-28T090953.945Z.wav', 'Abyan', '80.58%', '2023-05-28 16:09:56'),
(313, '2023-05-28T091002.709Z.wav', 'Ilmi', '65.80%', '2023-05-28 16:10:05'),
(314, '2023-05-28T091034.604Z.wav', 'Abyan', '83.50%', '2023-05-28 16:10:37'),
(315, '2023-05-28T091041.767Z.wav', 'Abyan', '98.51%', '2023-05-28 16:10:43'),
(316, '2023-05-28T091048.857Z.wav', 'Abyan', '83.42%', '2023-05-28 16:10:51'),
(317, '2023-05-28T091055.442Z.wav', 'Abyan', '94.64%', '2023-05-28 16:10:58'),
(318, '2023-05-28T091102.847Z.wav', 'Abyan', '88.77%', '2023-05-28 16:11:04'),
(319, '2023-05-28T091110.112Z.wav', 'Abyan', '94.54%', '2023-05-28 16:11:12'),
(320, '2023-05-28T091131.798Z.wav', 'Ilmi', '94.69%', '2023-05-28 16:11:34'),
(321, '2023-05-28T091139.167Z.wav', 'Ilmi', '68.53%', '2023-05-28 16:11:41'),
(322, '2023-05-28T091146.325Z.wav', 'Ilmi', '84.21%', '2023-05-28 16:11:48'),
(323, '2023-05-28T091153.001Z.wav', 'Abyan', '54.97%', '2023-05-28 16:11:55'),
(324, '2023-05-28T091200.284Z.wav', 'Ilmi', '79.84%', '2023-05-28 16:12:02'),
(325, '2023-05-28T091207.081Z.wav', 'Ilmi', '83.37%', '2023-05-28 16:12:09'),
(326, '2023-05-28T091214.056Z.wav', 'Ilmi', '66.79%', '2023-05-28 16:12:16'),
(327, '2023-05-28T091221.133Z.wav', 'Ilmi', '77.33%', '2023-05-28 16:12:23'),
(328, '2023-05-28T091227.908Z.wav', 'Abyan', '65.64%', '2023-05-28 16:12:30'),
(329, '2023-05-28T113338.866Z.wav', 'Ilmi', '80.61%', '2023-05-28 18:33:42'),
(330, '2023-05-28T113347.882Z.wav', 'Ilmi', '88.09%', '2023-05-28 18:33:50'),
(331, '2023-05-28T113355.383Z.wav', 'Ilmi', '96.75%', '2023-05-28 18:33:58'),
(332, '2023-05-28T113402.762Z.wav', 'Ilmi', '65.65%', '2023-05-28 18:34:05'),
(333, '2023-05-28T113410.247Z.wav', 'Ilmi', '79.83%', '2023-05-28 18:34:12'),
(334, '2023-05-28T113417.210Z.wav', 'Ilmi', '90.60%', '2023-05-28 18:34:19');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `admin`
--
ALTER TABLE `admin`
  ADD PRIMARY KEY (`id_admin`);

--
-- Indexes for table `dataset`
--
ALTER TABLE `dataset`
  ADD PRIMARY KEY (`id_dataset`),
  ADD KEY `kelas` (`label`);

--
-- Indexes for table `hyperparam`
--
ALTER TABLE `hyperparam`
  ADD PRIMARY KEY (`id_hyperparam`);

--
-- Indexes for table `kelas`
--
ALTER TABLE `kelas`
  ADD PRIMARY KEY (`id_kelas`);

--
-- Indexes for table `log_identifikasi`
--
ALTER TABLE `log_identifikasi`
  ADD PRIMARY KEY (`id_log`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `admin`
--
ALTER TABLE `admin`
  MODIFY `id_admin` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `dataset`
--
ALTER TABLE `dataset`
  MODIFY `id_dataset` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=473;

--
-- AUTO_INCREMENT for table `hyperparam`
--
ALTER TABLE `hyperparam`
  MODIFY `id_hyperparam` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `kelas`
--
ALTER TABLE `kelas`
  MODIFY `id_kelas` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=43;

--
-- AUTO_INCREMENT for table `log_identifikasi`
--
ALTER TABLE `log_identifikasi`
  MODIFY `id_log` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=335;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;