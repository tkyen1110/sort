"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import cv2
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

from metavision_sdk_core import EventBbox
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_core.event_io.py_reader import EventDatReader
from metavision_core.event_io.events_iterator import EventsIterator
from metavision_core.event_io import EventNpyReader

np.random.seed(0)

def nms(box_events, scores, iou_thresh=0.5):
    """NMS on box_events

    Args:
        box_events (np.ndarray): nx1 with dtype EventBbox, the sorting order of those box is used as a
            a criterion for the nms.
        scores (np.ndarray): nx1 dtype of plain dtype, needs to be argsortable.
        iou_thresh (float): if two boxes overlap with more than `iou_thresh` (intersection over union threshold)
            with each other, only the one with the highest criterion value is kept.

    Returns:
        keep (np.ndarray): Indices of the box to keep in the input array.
    """
    x1 = box_events['x']
    y1 = box_events['y']
    x2 = box_events['x'] + box_events['w']
    y2 = box_events['y'] + box_events['h']

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return sorted(keep)

def linear_assignment(cost_matrix):
	try:
		import lap
		_, x, y = lap.lapjv(cost_matrix, extend_cost=True)
		return np.array([[y[i],i] for i in x if i >= 0]) #
	except ImportError:
		from scipy.optimize import linear_sum_assignment
		x, y = linear_sum_assignment(cost_matrix)
		return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
	"""
	From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
	"""
	bb_gt = np.expand_dims(bb_gt, 0)
	bb_test = np.expand_dims(bb_test, 1)
	
	xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
	yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
	xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
	yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
	w = np.maximum(0., xx2 - xx1)
	h = np.maximum(0., yy2 - yy1)
	wh = w * h
	o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
		+ (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
	return(o)  


def convert_bbox_to_z(bbox):
	"""
	Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
		[x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
		the aspect ratio
	"""
	w = bbox[2] - bbox[0]
	h = bbox[3] - bbox[1]
	x = bbox[0] + w/2.
	y = bbox[1] + h/2.
	s = w * h    #scale is just area
	r = w / float(h)
	return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
	"""
	This class represents the internal state of individual tracked objects observed as bbox.
	"""
	count = 0
	def __init__(self,bbox):
		"""
		Initialises a tracker using initial bounding box.
		"""
		#define constant velocity model
		self.kf = KalmanFilter(dim_x=7, dim_z=4) 
		self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
		self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

		self.kf.R[2:,2:] *= 10.
		self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
		self.kf.P *= 10.
		self.kf.Q[-1,-1] *= 0.01
		self.kf.Q[4:,4:] *= 0.01

		self.kf.x[:4] = convert_bbox_to_z(bbox)
		self.time_since_update = 0
		self.id = KalmanBoxTracker.count
		KalmanBoxTracker.count += 1
		self.history = []
		self.bboxes_center = []
		self.bboxes_density = []
		self.hits = 0
		self.hits_old = -1
		self.hit_streak = 0
		self.age = 0

	def update(self, bbox, ev_height, ev_width):
		"""
		Updates the state vector with observed bbox.
		"""
		self.time_since_update = 0
		self.history = []
		self.hits += 1
		self.hit_streak += 1
		x1,y1,x2,y2, confidence = bbox
		x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
		self.bboxes_center.append(np.array([(x1+x2)//2, (y1+y2)//2]))

		x1 = np.clip(x1, 0, ev_width)
		y1 = np.clip(y1, 0, ev_height)
		x2 = np.clip(x2, 0, ev_width)
		y2 = np.clip(y2, 0, ev_height)
		w = x2 - x1 
		h = y2 - y1
		density = np.sum(bmap[y1:y2, x1:x2]) / (w*h)
		self.bboxes_density.append(density)
   
		self.kf.update(convert_bbox_to_z(bbox))

	def predict(self):
		"""
		Advances the state vector and returns the predicted bounding box estimate.
		"""
		if((self.kf.x[6]+self.kf.x[2])<=0):
			self.kf.x[6] *= 0.0
		self.kf.predict()
		self.age += 1
		if(self.time_since_update>0):
			self.hit_streak = 0
		self.time_since_update += 1

		if self.hits_old != self.hits:
			self.history.append(convert_x_to_bbox(self.kf.x))
			self.hits_old = self.hits
		return self.history[-1]

	def get_state(self):
		"""
		Returns the current bounding box estimate.
		"""
		if len(self.history) == 0:
			return convert_x_to_bbox(self.kf.x)
		else:
			return self.history[-1]


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
	"""
	Assigns detections to tracked object (both represented as bounding boxes)

	Returns 3 lists of matches, unmatched_detections and unmatched_trackers
	"""
	if(len(trackers)==0):
		return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

	iou_matrix = iou_batch(detections, trackers)

	if min(iou_matrix.shape) > 0:
		a = (iou_matrix > iou_threshold).astype(np.int32)
		if a.sum(1).max() == 1 and a.sum(0).max() == 1:
			matched_indices = np.stack(np.where(a), axis=1)
		else:
			matched_indices = linear_assignment(-iou_matrix)
	else:
		matched_indices = np.empty(shape=(0,2))

	unmatched_detections = []
	for d, det in enumerate(detections):
		if(d not in matched_indices[:,0]):
			unmatched_detections.append(d)
	unmatched_trackers = []
	for t, trk in enumerate(trackers):
		if(t not in matched_indices[:,1]):
			unmatched_trackers.append(t)

	#filter out matched with low IOU
	matches = []
	for m in matched_indices:
		if(iou_matrix[m[0], m[1]]<iou_threshold):
			unmatched_detections.append(m[0])
			unmatched_trackers.append(m[1])
		else:
			matches.append(m.reshape(1,2))
	if(len(matches)==0):
		matches = np.empty((0,2),dtype=int)
	else:
		matches = np.concatenate(matches,axis=0)

	return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
	def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
		"""
		Sets key parameters for SORT
		"""
		self.max_age = max_age
		self.min_hits = min_hits
		self.iou_threshold = iou_threshold
		self.trackers = []
		self.frame_count = 0

	def update(self, dets=np.empty((0, 5)), bmap=None, ev_height=0, ev_width=0):
		"""
		Params:
		dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
		Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
		Returns the a similar array, where the last column is the object ID.

		NOTE: The number of objects returned may differ from the number of detections provided.
		"""
		self.frame_count += 1
		# get predicted locations from existing trackers.
		trks = np.zeros((len(self.trackers), 5))
		to_del = []
		ret = []
		discard = []
		for t, trk in enumerate(trks):
			pos = self.trackers[t].predict()[0]
			trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
			if np.any(np.isnan(pos)):
				to_del.append(t)
		trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
		for t in reversed(to_del):
			self.trackers.pop(t)
		matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

		# update matched trackers with assigned detections
		for m in matched:
			self.trackers[m[1]].update(dets[m[0], :], ev_height, ev_width)

		# create and initialise new trackers for unmatched detections
		for i in unmatched_dets:
			trk = KalmanBoxTracker(dets[i,:])
			self.trackers.append(trk)
		i = len(self.trackers)
		for trk in reversed(self.trackers):
			d = trk.get_state()[0]
			x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
			x1 = np.clip(x1, 0, ev_width)
			y1 = np.clip(y1, 0, ev_height)
			x2 = np.clip(x2, 0, ev_width)
			y2 = np.clip(y2, 0, ev_height)
			w = x2 - x1 
			h = y2 - y1
			density = np.sum(bmap[y1:y2, x1:x2]) / (w*h)
			count = 0
			if (
       			((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)) or
 
			    ((trk.hits == trk.hits_old) and len(trk.bboxes_density)>=2 and np.max(trk.bboxes_density)>0.5 and trk.bboxes_density[-1]<0.2 and
       			 (np.abs(trk.bboxes_center[-1][0] - trk.bboxes_center[-2][0])<=3) and (np.abs(trk.bboxes_center[-1][1] - trk.bboxes_center[-2][1])<=3))
       		   ):
   			#    ((trk.hits == trk.hits_old) and len(trk.bboxes_center) >=2 and np.abs(trk.bboxes_center[-1][0] - trk.bboxes_center[-2][0])<=3 and
          	# 	np.abs(trk.bboxes_center[-1][1] - trk.bboxes_center[-2][1])<=3) or 
         	#    (len(trk.bboxes_center)==0 and density>=0.2)):
				ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
				count = count + 1
				if len(trk.bboxes_center) == 0:
					print("O", d, density, len(trk.bboxes_center), trk.id+1)
				else:
					print("O", d, density, len(trk.bboxes_center), trk.bboxes_center[0], trk.bboxes_center[-1], trk.bboxes_density[0], trk.bboxes_density[-1], trk.id+1)
    
			i -= 1
			# remove dead tracklet
			# if (trk.time_since_update > self.max_age):
			if (
       			((trk.time_since_update > self.max_age) and w*h < 2500) or 
       
       			((trk.hits == trk.hits_old) and len(trk.bboxes_center)<5) or
          
				((trk.hits == trk.hits_old) and len(trk.bboxes_density)>0 and trk.bboxes_density[-1]>0.4) or
    
       			((trk.hits == trk.hits_old) and len(trk.bboxes_density) >=2 and np.max(trk.bboxes_density)>0.5 and 
           		 (trk.bboxes_density[-1]>=0.2 or np.abs(trk.bboxes_center[-1][0] - trk.bboxes_center[-2][0])>3 or
          		  np.abs(trk.bboxes_center[-1][1] - trk.bboxes_center[-2][1])>3)) or 
          
       			(len(trk.bboxes_center)==0 and density<0.1)
          	   ):
				self.trackers.pop(i)
				discard.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
				count = count + 1
				if len(trk.bboxes_center) == 0:
					print("X", d, density, len(trk.bboxes_center), trk.id+1)
				else:
					print("X", d, density, len(trk.bboxes_center), trk.bboxes_center[0], trk.bboxes_center[-1], trk.bboxes_density[0], trk.bboxes_density[-1], trk.id+1)

			if count == 0:
				if len(trk.bboxes_center) == 0:
					print("X", d, density, len(trk.bboxes_center), trk.id+1)
				else:
					print("X", d, density, len(trk.bboxes_center), trk.bboxes_center[0], trk.bboxes_center[-1], trk.bboxes_density[0], trk.bboxes_density[-1], trk.id+1)
     
		print("= "*50)
		if(len(ret)>0):
			return np.concatenate(ret)
		return np.empty((0,5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	# all train
	args = parse_args()
	display = args.display
	seq_path = args.seq_path
	phase = args.phase
	total_time = 0.0
	total_frames = 0
	downsample = 1

	evaluation_dir = "/home/tkyen/opencv_practice/data_3/Gen4_Automotive_event_cube_paper/result_vanilla_ssd_level_5/evaluation_epoch_31"
	event_dir = "/home/tkyen/opencv_practice/data_1/Gen4_Automotive/test_dat"
	output_dir = os.path.join(evaluation_dir, "dt_SORT")
	os.makedirs(output_dir, exist_ok=True)

	gt_label_dir = os.path.join(evaluation_dir, "gt")
	dt_label_dir = os.path.join(evaluation_dir, "dt")
	pattern = os.path.join(dt_label_dir, '*_bbox.npy')
	delta_t = 50000

	colors_hsv = np.uint8([[[0,255,255], [20,255,255], [40,255,255], [60,255,255], [80,255,255], [100,255,255], [120,255,255], [140,255,255]]])
	colors_bgr = cv2.cvtColor(colors_hsv, cv2.COLOR_HSV2BGR).squeeze()
	colors = colors_bgr.tolist()

	for dt_label_path in glob.glob(pattern):
		seq_base = os.path.basename(dt_label_path)
		seq = seq_base.replace("_bbox.npy","")
		if seq not in ["moorea_2019-02-19_001_td_1220500000_1280500000",
					   "moorea_2019-02-19_003_td_1586500000_1646500000",
                 	   "moorea_2019-02-19_005_td_915500000_975500000",
					   "moorea_2019-02-19_005_td_1159500000_1219500000",
        			   "moorea_2019-02-19_005_td_1220500000_1280500000",
              		   "moorea_2019-02-21_000_td_2440500000_2500500000",
                   	   "moorea_2019-02-21_000_td_2501500000_2561500000"]:
			continue
		event_path = os.path.join(event_dir, seq_base.replace("_bbox.npy","_td.dat"))
		gt_label_path = os.path.join(gt_label_dir, seq_base)
		output_path = os.path.join(output_dir, seq_base)

  		# create instance of the SORT tracker
		KalmanBoxTracker.count = 0
		mot_tracker = Sort(	max_age=args.max_age, 
							min_hits=args.min_hits,
							iou_threshold=args.iou_threshold )

		event_dat = EventDatReader(event_path)
		event_dat.seek_time(0)
		ev_height, ev_width = event_dat.get_size()
		ev_height = ev_height >> downsample
		ev_width = ev_width >> downsample

		gt_label = EventNpyReader(gt_label_path)
		gt_label.seek_time(0)

		dt_label = EventNpyReader(dt_label_path)
		dt_label.seek_time(0)

		# VideoWriter
		fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		frame_rate = 10
		video_writer = cv2.VideoWriter(os.path.join(output_dir, '{}.mp4'.format(seq)), fourcc, frame_rate, (ev_width*2, ev_height*2))
		ev_dtype = {'names':['x','y','p','t'], 'formats':['<u2','<u2','<i2','<i8'], 'offsets':[0,2,4,8], 'itemsize':16}
    
		print("Processing %s."%(seq))
		for frame in range(int(60e6//delta_t)):
			print('{} ms'.format(frame*delta_t//1000))
			frame += 1 #detection and frame numbers begin at 1

			events = event_dat.load_delta_t(delta_t)
			events['x'] = events['x'] >> downsample
			events['y'] = events['y'] >> downsample
			bmap = np.zeros((ev_height, ev_width), dtype=bool)
			bmap[events['y'], events['x']] = 1

			image_all = np.zeros((ev_height*2, ev_width*2, 3), dtype=np.uint8)
			BaseFrameGenerationAlgorithm.generate_frame(events, image_all[:ev_height, :ev_width, :])
			image_all[:ev_height, ev_width:, :] = image_all[:ev_height, :ev_width, :].copy()
			image_all[ev_height:, :ev_width, :] = image_all[:ev_height, :ev_width, :].copy()
			image_all[ev_height:, ev_width:, :] = image_all[:ev_height, :ev_width, :].copy()
			# top left
			cv2.putText(image_all, 'GT',
						(10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
			cv2.putText(image_all, '{} ms / {} ms'.format(frame*delta_t//1000, (frame+1)*delta_t//1000), 
						(ev_width-200, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
			# bottom left
			cv2.putText(image_all, 'GT',
						(10, ev_height+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
			cv2.putText(image_all, '{} ms / {} ms'.format(frame*delta_t//1000, (frame+1)*delta_t//1000), 
						(ev_width-200, ev_height+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
			# top right
			cv2.putText(image_all, 'DT',
						(ev_width+10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
			# bottom right
			cv2.putText(image_all, 'Tracking',
						(ev_width+10, ev_height+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
			image_all = cv2.line(image_all, (0,ev_height), (ev_width*2, ev_height), (0,0,255), 1)
			image_all = cv2.line(image_all, (ev_width,0), (ev_width,ev_height*2), (0,0,255), 1)

			gt_boxes = gt_label.load_delta_t(delta_t=delta_t)
			gt_boxes = gt_boxes[nms(gt_boxes, gt_boxes['class_confidence'], iou_thresh=0.5)]

			dt_boxes = dt_label.load_delta_t(delta_t=delta_t)
			dt_boxes = dt_boxes[dt_boxes['class_confidence']>=0.5]
			dt_boxes = dt_boxes[nms(dt_boxes, dt_boxes['class_confidence'], iou_thresh=0.5)]
			# dt_boxes.dtype = ['t','x','y','w','h','class_id','track_id','class_confidence']
			for gt_box in gt_boxes:
				if len(gt_box) == 9:
					t, x, y, w, h, class_id, confidence, track_id, invalid = gt_box
					x, y, w, h = int(x), int(y), int(w), int(h)
					if invalid == False:
						cv2.rectangle(image_all, (x, y), (x+w, y+h), colors[class_id], 1)
						cv2.rectangle(image_all, (x, ev_height+y), (x+w, ev_height+y+h), colors[class_id], 1)
				else:
					t, x, y, w, h, class_id, track_id, confidence = gt_box
					x, y, w, h = int(x), int(y), int(w), int(h)
					cv2.rectangle(image_all, (x, y), (x+w, y+h), colors[class_id], 1)
					cv2.rectangle(image_all, (x, ev_height+y), (x+w, ev_height+y+h), colors[class_id], 1)

			for dt_box in dt_boxes:
				t, x, y, w, h, class_id, track_id, confidence = dt_box
				x, y, w, h = int(x), int(y), int(w), int(h)
				# cv2.putText(image_all, str(round(confidence, 3)), (ev_width+x, y-6), cv2.FONT_HERSHEY_TRIPLEX, 0.6, colors[class_id], 1, cv2.LINE_AA)
				# cv2.putText(image_all, str("[{}, {}]".format(int(x+w//2), int(y+h//2))), (ev_width+x+40, y-6), cv2.FONT_HERSHEY_TRIPLEX, 0.6, colors[0], 1, cv2.LINE_AA)
				cv2.rectangle(image_all, (ev_width+x, y), (ev_width+x+w, y+h), colors[class_id], 1)

			dets = np.concatenate((dt_boxes['x'][:, None], dt_boxes['y'][:, None], dt_boxes['w'][:, None], dt_boxes['h'][:, None], dt_boxes['class_confidence'][:, None]), axis=1)
			dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
			# dets = np.array([[x1, y1, x2, y2, confidence], 
			#                  [...]])
			total_frames += 1

			start_time = time.time()
			trackers = mot_tracker.update(dets, bmap, ev_height, ev_width)
			cycle_time = time.time() - start_time
			total_time += cycle_time

			for d in trackers:
				x, y, x2, y2, track_id = d[0], d[1], d[2], d[3], d[4]
				w = x2 -x
				h = y2 -y
				x, y, w, h = int(x), int(y), int(w), int(h)
				cv2.putText(image_all, str(int(track_id)), (ev_width+x, ev_height+y-6), cv2.FONT_HERSHEY_TRIPLEX, 0.6, colors[0], 1, cv2.LINE_AA)
				# cv2.putText(image_all, str("[{}, {}]".format(int((x+x2)//2), int((y+y2)//2))), (ev_width+x+40, ev_height+y-6), cv2.FONT_HERSHEY_TRIPLEX, 0.6, colors[0], 1, cv2.LINE_AA)
				cv2.rectangle(image_all, (ev_width+x, ev_height+y), (ev_width+x+w, ev_height+y+h), colors[0], 1)
				# print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)

			video_writer.write(image_all)
		video_writer.release()
	print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

	if(display):
		print("Note: to get real runtime results run without the option: --display")
